//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro_tasks.cpp
//! \brief functions that control NBody tasks stored in tasklists in MeshBlockPack

#include <map>
#include <memory>
#include <string>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "hydro/hydro.hpp"

#include "nbody/nbody.hpp"

namespace nbody {

void NBody::AssembleNBodyTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  
  TaskID none(0);

  // coarse timestep execution
  // id.reduce_mesh = tl["after_timeintegrator"]->AddTask(&NBody::ReduceParentMesh, this, none);
  // id.reduce_meshes = tl["after_timeintegrator"]->AddTask(&NBody::ReduceAllMeshes, this, id.reduce_mesh);
  // id.integrate = tl["after_timeintegrator"]->AddTask(&NBody::Integrate, this, id.reduce_meshes);
  // id.scatter = tl["after_timeintegrator"]->AddTask(&NBody::Scatter, this, id.integrate);
  // id.calc_dt = tl["after_timeintegrator"]->AddTask(&NBody::NewTimeStep, this, id.scatter);

  // fine timestep execution
  id.initrk = tl["stagen"]->AddTask(&NBody::InitRK, this, none);
  id.flux   = tl["stagen"]->AddTask(&NBody::Fluxes, this, id.initrk);
  id.rkupdt = tl["stagen"]->AddTask(&NBody::RKUpdate, this, id.flux);
  id.calc_dt  = tl["stagen"]->AddTask(&NBody::NewTimeStep, this, id.scatter);

  return;
}

// taskstatus wrapper to CalcTimeStep
TaskStatus NBody::NewTimeStep(Driver *pdrive, int stage) {
  
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  // symmetrise nbody timestep with previous value
  const Real dt_current = CalcTimeStep();
  const Real dt_sqr = dt_current * dt_current;
  dt_new = dt_sqr / dt_old;

  return TaskStatus::complete;
}

// collect forcing by hydro on nbody state on THIS rank
TaskStatus NBody::ReduceParentMesh(Driver *pdrive, int stage) {

  // Step 1: If running without backreaction, skip summation
  if (!sum_backreaction) return TaskStatus::complete;

  // Step 2: only run commincation on ONE mb_pack per rank
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // Step 3: Init pack sum register as zero 
  Kokkos::deep_copy(delta_this_mesh.view_host(), 0.0);

  // Step 4: Collect updates across mb_packs on this rank 
  for (int mbp_id = 0; mbp_id < pmy_pack->pmesh->nmb_packs_thisrank; mbp_id++) {
    // enforce device->host sync for MeshBlockPack
    pmy_pack->pmesh->pmb_pack[mbp_id].pnbody->delta_this_pack.template modify<DevExeSpace>(); // TODO: shift this to source term?
    pmy_pack->pmesh->pmb_pack[mbp_id].pnbody->delta_this_pack.template sync<HostMemSpace>();
    // sum device-synchronised pack data into mesh data
    for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < NVAR_BACK; i++) {
        delta_this_mesh.h_view(n, i) += pmy_pack->pmesh->pmb_pack[mbp_id].pnbody->delta_this_pack.h_view(n, i);
      } // end NVAR_BACK loop
    } // end n loop
  } // end mb_pack loop

  if (verbose) {
    std::cout << "Collected back reaction across " << pmy_pack->pmesh->nmb_packs_thisrank 
              << " MeshBlockPack(s) on rank " << global_variable::my_rank << std::endl;
  }

  return TaskStatus::complete;
}

// collect forcing by hydro on nbody state on ALL ranks
TaskStatus NBody::ReduceAllMeshes(Driver *pdrive, int stage) {

  // Step 1: If running without backreaction, skip
  if (!sum_backreaction) return TaskStatus::complete;

  // Step 2: only run commincation on ONE mb_pack per rank
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // Step 3: sum delta_pack_sum across ranks
  Kokkos::deep_copy(delta_all_meshes.view_host(), 0.0); // set to zero pre-reduction
#if MPI_PARALLEL_ENABLED
  // template: MPI_Reduce(write_addr, read_addr, data_count, data_type, operation, scope)
  MPI_Reduce(&delta_all_meshes.view_host(), &delta_this_mesh.view_host(), NVAR_BACK, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif

  // Step 4: convert sum of deltas into rates 
  for (int n = 0; n < num_nbody; n++) {
    for (int i = 0; i < NVAR_REG; i++) {
      if (i == 0) { // mdot = dm / dt
        delta_all_meshes.h_view(n, i) = delta_all_meshes.h_view(n, i) / pmy_pack->pmesh->dt; 
      } else { // vdot = dp / (m * dt)
        delta_all_meshes.h_view(n, i) = delta_all_meshes.h_view(n, i) / (nbody_data.h_view(n, M_DATA) * pmy_pack->pmesh->dt);
      }
    }
  }
  
  if (verbose) {
    std::cout << "Reduced back reaction across " << global_variable::nranks << " rank(s)"
              << "from rank " << global_variable::my_rank << std::endl;
  }

  return TaskStatus::complete;
}

// propogate nbody state forward in time using RK4
TaskStatus NBody::Integrate(Driver *pdrive, int stage) {

  // Step 1: only perform integration on ONE mb_pack on ONE rank
  if (global_variable::my_rank != 0) return TaskStatus::complete;
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // Step 2: perform RK4 integration of nbody state

  // integrate on coarse timestep dt (NOT beta_dt)
  const Real dt = pmy_pack->pmesh->dt;
  const Real dt_over_6 = dt / 6.0;

  // package data into scratch registers
  // i runs over m,3x,3vx...
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < NVAR_REG; i++) {
          _y_init.h_view(n, i) = nbody_data.h_view(n, i);
      } // end i
  } // end n

  // compute k coefficients for RK4 substeps
  
  // step 1
  EvaluateF(_y_init, _k_sub);
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < NVAR_REG; i++) {
          _y_sub.h_view(n, i) = _y_init.h_view(n, i) + 0.5 * dt * _k_sub.h_view(n, i);
          _y_ret.h_view(n, i) = _y_init.h_view(n, i) + dt_over_6 * _k_sub.h_view(n, i);
      } // end i
  } // end n
  
  // step 2
  EvaluateF(_y_sub, _k_sub);
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < NVAR_REG; i++) {
          _y_sub.h_view(n, i) = _y_init.h_view(n, i) + 0.5 * dt * _k_sub.h_view(n, i);
          _y_ret.h_view(n, i) += 2.0 * dt_over_6 * _k_sub.h_view(n, i);
      } // end i
  } // end n

  // step 3
  EvaluateF(_y_sub, _k_sub);
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < NVAR_REG; i++) {
          _y_sub.h_view(n, i) = _y_init.h_view(n, i) + dt * _k_sub.h_view(n, i);
          _y_ret.h_view(n, i) += 2.0 * dt_over_6 * _k_sub.h_view(n, i);
      } // end i
  } // end n

  // step 4
  EvaluateF(_y_sub, _k_sub);
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < NVAR_REG; i++) {
          _y_ret.h_view(n, i) += dt_over_6 * _k_sub.h_view(n, i);
      } // end i
  } // end n

  // update main register
  // no need to empty sub-step registers, all overwritten in next
  for (int n = 0; n < num_nbody; n++) {
    for (int i = 0; i < NVAR_REG; i++) {
        nbody_data.h_view(n, i) = _y_ret.h_view(n, i);
      } // end i
  }

  if (verbose) {
    std::cout << "Completed RK4 integration on MeshBlockPack " << pmy_pack->pmesh->nmb_packs_thisrank
              << " on rank " << global_variable::my_rank << std::endl; 
  }

  return TaskStatus::complete;
}

// register initialistaion for finestep integraton
TaskStatus NBody::InitRK(Driver *pdrive, int stage) {
  
  // only run this task on the principle meshblock pack (rank0, mbpid=0)
  if (global_variable::my_rank != 0) return TaskStatus::complete;
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;
  
  if (stage == 1) {
    Kokkos::deep_copy(HostMemSpace(), nbody_data1, nbody_data.view_host());
  } else {
    if (pdrive->integrator == "rk4") {
      // parallel loop to update y1 with y0 at later stages, only for rk4
      Real &delta = pdrive->delta[stage-1];
      for (int n = 0; n < num_nbody; n++) {
        for (int i = 0; i < NVAR_REG; i++) {
          nbody_data1(n, i) += delta * nbody_data.h_view(n, i);
        } // end i
      } // end n
    } // end rk4
  } // end stage check
  return TaskStatus::complete;
}

// compute flux for nbody state
TaskStatus NBody::Fluxes(Driver *pdrive, int stage) {
  
  // evaluate time evoluton of the NBody state nbody_flux
  // only run this task on the principle meshblock pack (rank0, mbpid=0)
  if (global_variable::my_rank != 0) return TaskStatus::complete;
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // Step 1: Collect nbody deltas across system (WARNING: currently safe only for single MeshBlockPack)
  // Convert deltas into rates (e.g. dm into mdot), averaging over fine timestep
  // TODO: add MPI comm stept to collect over ranks here
  Real beta_dt = (pdriver->beta[stage-1])*(pmy_pack->pmesh->dt);
  for (int n = 0; n < num_nbody; n++) {
    for (int i = 0; i < NVAR_REG; i++) {
      if (i == 0) { // mdot = dm / dt
        delta_all_meshes.h_view(n, i) = delta_this_pack.h_view(n, i) / beta_dt;
      } else { // vdot = dp / (m * dt)
        delta_all_meshes.h_view(n, i) = delta_this_pack.h_view(n, i) / (nbody_data.h_view(n, M_DATA) * beta_dt);
      }
    }
  }

  // Step 2: Set nbody flux to zero for entire register
  Kokkos::deep_copy(HostMemSpace(), nbody_flux, 0.0);

  // Step 3: Compute flux for each body in class
  for (int n = 0; n < num_nbody; n++) {
    // register now contains mdot, not delta m
    nbody_flux(n, MDOT_REG) = (inc_backreaction) ? delta_all_meshes.h_view(n, DM_BACK) : 0.0;

    // dot(x) = v
    nbody_flux(n, XDOT_REG) = nbody_data.h_view(n, VX_REG);
    nbody_flux(n, YDOT_REG) = nbody_data.h_view(n, VY_REG);
    nbody_flux(n, ZDOT_REG) = nbody_data.h_view(n, VZ_REG);
    
    // dot(v) = dot(p) / m (use m from start of integration)
    // registers now contain accelerations, not momentum changes
    nbody_flux(n, VXDOT_REG) = (inc_backreaction) ? delta_all_meshes.h_view(n, DPX_GRAV_BACK) + delta_all_meshes.h_view(n, DPX_ACC_BACK) : 0.0;
    nbody_flux(n, VYDOT_REG) = (inc_backreaction) ? delta_all_meshes.h_view(n, DPY_GRAV_BACK) + delta_all_meshes.h_view(n, DPY_ACC_BACK) : 0.0;
    nbody_flux(n, VZDOT_REG) = (inc_backreaction) ? delta_all_meshes.h_view(n, DPZ_GRAV_BACK) + delta_all_meshes.h_view(n, DPZ_ACC_BACK) : 0.0;

    // all later indices of nbody_flux left as ZERO

    // add acceleraton by mutual nbody gravity
    for (int m = 0; m < num_nbody; m++) {
      if (m == n) continue; // no self-gravity
        
      // extract spatial seperation
      const Real dx = nbody_data.h_view(n, X_REG) - nbody_data.h_view(m, X_REG);
      const Real dy = nbody_data.h_view(n, Y_REG) - nbody_data.h_view(m, Y_REG);
      const Real dz = nbody_data.h_view(n, Z_REG) - nbody_data.h_view(m, Z_REG);
      Real r_sqr = SQR(dx) + SQR(dy) + SQR(dz);

      // branch if PostNewtonian forcing to be included
      if (!inc_pn) {
        // compute Newtnonina pairwise acceleration
        const Real g_fac = _G * nbody_data.h_view(m, M_REG) / (r_sqr * std::sqrt(r_sqr));
        
        // decompose acceleration and update
        nbody_flux(n, VXDOT_REG) -= g_fac * dx;
        nbody_flux(n, VYDOT_REG) -= g_fac * dy;
        nbody_flux(n, VZDOT_REG) -= g_fac * dz; 
      } else {
        // include additional expansion terms up to 2.5PN (WIP)
        // formulaism from Eq 203 of "Gravitational Radiation from Post-Newtonian Sources 
        // and Inspiralling Compact Binaries" Blanchet 2014
        // notation switch from (1,2)->(n,m)


        // label masses
        Real m1 = nbody_data.h_view(n, M_REG);
        Real m2 = nbody_data.h_view(m, M_REG);

        // extract velocity terms
        const Real dvx = nbody_data.h_view(n, VX_REG) - nbody_data.h_view(m, VX_REG);
        const Real dvy = nbody_data.h_view(n, VY_REG) - nbody_data.h_view(m, VY_REG);
        const Real dvz = nbody_data.h_view(n, VZ_REG) - nbody_data.h_view(m, VZ_REG);
        Real v_sqr = SQR(dvx) + SQR(dvy) + SQR(dvz);
        Real v_dot = nbody_data.h_view(n, VX_REG) * nbody_data.h_view(m, VX_REG) + 
                      nbody_data.h_view(n, VY_REG) * nbody_data.h_view(m, VY_REG) +
                      nbody_data.h_view(n, VZ_REG) * nbody_data.h_view(m, VZ_REG);

        // compute unit directions
        Real r = Kokkos::sqrt(r_sqr);
        Real inv_r = 1.0 / r;
        Real inv_r_sqr = SQR(inv_r);
        Real n_x = dx * inv_r;
        Real n_y = dy * inv_r;
        Real n_z = dz * inv_r;

        // TODO: add unit conversions here, including for _G

        // 0th order (Newtonian)
        const Real newtonian_fac = -_G * m2 * inv_r_sqr;
        Real a_x = newtonian_fac * n_x;
        Real a_y = newtonian_fac * n_y;
        Real a_z = newtonian_fac * n_z;

        // TODO: add higher order terms here

        // convert back to code units and 
        nbody_flux(n, VXDOT_REG) += a_x / unit_A;
        nbody_flux(n, VYDOT_REG) += a_y / unit_A;
        nbody_flux(n, VZDOT_REG) += a_z / unit_A;
      } // end pn branch
    } // end m loop
  } // end n loop

  // Step 4: Cleanup collection registers for next finetimestep
  Kokkos::deep_copy(delta_all_meshes.view_host(), 0.0);
  Kokkos::deep_copy(HostMemSpace(), delta_this_mesh, 0.0); // TODO: currently unused as no MPI comm
  Kokkos::deep_copy(HostMemSpace(), delta_this_pack, 0.0);

  // Step 5: Enforce update of register state on device for DualArray delta_this_pack
  delta_this_pack.template modify<HostMemSpace>();
  delta_this_pack.template sync<DevExeSpace>();

  // Step 5: Report, if flagged
  if (verbose) {
    std::cout << "Completed stage " << stage << " of NBody::Fluxes on MeshBlockPack " << pmy_pack->pmesh->nmb_packs_thisrank
              << " on rank " << global_variable::my_rank << std::endl; 
  }

  return TaskStatus::complete;
}

// propogate nbody state forward by a single fine timestep
TaskStatus NBody::RKUpdate(Driver *pdrive, int stage) {

  // load integration weights from general time integrator
  Real &gam0 = pdrive->gam0[stage-1];
  Real &gam1 = pdrive->gam1[stage-1];
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);

  // Step 1: only perform integration on ONE mb_pack on ONE rank
  if (global_variable::my_rank != 0) return TaskStatus::complete;
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // Step 2: bump nbody register to next fine timestep
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < NVAR_REG; i++) { // skip iteration over "static" indices beyond NVAR_REG
          nbody_data.h_view(n, i) = gam0 * nbody_data + gam1 * nbody_data1 + beta_dt * nbody_data_flux;
      } // end i
  } // end n

  // Step 3: enforce update of nbody state on device for DualArray registers
  nbody_data.template modify<HostMemSpace>();
  nbody_data.template sync<DevExeSpace>();

  if (verbose) {
    std::cout << "Completed stage " << stage << " of NBody integration on MeshBlockPack " << pmy_pack->pmesh->nmb_packs_thisrank
              << " on rank " << global_variable::my_rank << std::endl; 
  }

  return TaskStatus::complete;
}

// scatter nbody state from rank 0 to all ranks
TaskStatus NBody::Scatter(Driver *pdrive, int stage) {

  // Step 1: scatter from and to ONE mb_pack per rank
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // Step 2: scatter nbody state from rank 0 to all
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(nbody_data.view_host(), NVAR_DATA * num_nbody, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
#endif

  // Step 3: scatter nbody state from this mb_pack to all on rank
  for (int mbp_id = 1; mbp_id < pmy_pack->pmesh->nmb_packs_thisrank; mbp_id++) {
    Kokkos::deep_copy(pmy_pack->pmesh->pmb_pack[mbp_id].pnbody->nbody_data.view_host(), nbody_data.view_host());
  } // end mb_pack loop

  // Step 4: reset registers and force update of device state on ALL mb_packs
  for (int mbp_id = 0; mbp_id < pmy_pack->pmesh->nmb_packs_thisrank; mbp_id++) {
    // wipe source-accesible backreaction register for next step
    Kokkos::deep_copy(pmy_pack->pmesh->pmb_pack[mbp_id].pnbody->delta_this_pack.view_host(), 0.0);
    // sync updates to device
    pmy_pack->pmesh->pmb_pack[mbp_id].pnbody->nbody_data.template modify<HostMemSpace>();
    pmy_pack->pmesh->pmb_pack[mbp_id].pnbody->delta_this_pack.template modify<HostMemSpace>();
    pmy_pack->pmesh->pmb_pack[mbp_id].pnbody->nbody_data.template sync<DevExeSpace>();
    pmy_pack->pmesh->pmb_pack[mbp_id].pnbody->delta_this_pack.template sync<DevExeSpace>();
  } // end mb_pack loop
  
  if (verbose) {
    std::cout << "Forced Scatter from root MeshBlockPack " 
              << " on rank " << global_variable::my_rank << std::endl; 
  }

  return TaskStatus::complete;
}


} // end namespace nbody