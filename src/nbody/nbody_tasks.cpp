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

//----------------------------------------------------------------------------------------
//! \fn  void NBody::AssembleNBodyTasks
//! \brief Adds nbody tasks to appropriate task lists used by time integrators.
//! Called by MeshBlockPack::AddPhysics() function directly after NBody constructor.
void NBody::AssembleNBodyTasks(std::map<std::string, std::shared_ptr<TaskList>> tl) {
  
  TaskID none(0);

  // fine timestep execution
  // id.initrk   = tl["stagen"]->AddTask(&NBody::InitRK, this, none);
  // id.flux     = tl["stagen"]->AddTask(&NBody::Fluxes, this, id.initrk);
  // id.rkupdt   = tl["stagen"]->AddTask(&NBody::RKUpdate, this, id.flux);
  // id.send     = tl["stagen"]->AddTask(&NBody::Send, this, id.rkupdt);
  // id.newdt    = tl["stagen"]->AddTask(&NBody::NewTimeStep, this, id.send);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::InitRK
//! \brief Simple task list function that computes the maximum time step for stable nbody
//! integration and symmtetrises it with the previous stable timestep. This time step is 
//! enfactored in the Mesh::NewTimeStep() function at the end of the time step
TaskStatus NBody::NewTimeStep(Driver *pdrive, int stage) {
  
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete; // only execute last stage
  }

  // symmetrise nbody timestep with previous value
  const Real dt_current = CalcTimeStep();
  const Real dt_sqr = dt_current * dt_current;
  dt_new = dt_sqr / dt_old;

  // check against diffusive timescale
  if (alpha != 0.0) {
    dt_new = std::max(dt_new, dt_visc);
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::InitRK
//! \brief Simple task list function that copies nbody_data --> nbody_data1 in first 
//! stage. Extended to handle RK register logic at given stage
TaskStatus NBody::InitRK(Driver *pdrive, int stage) {
  
  // only run this task on the principle meshblock pack (rank0, mbpid=0)
  // if (global_variable::my_rank != 0) return TaskStatus::complete;
  // if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  if (stage == 1) {
    // copy by element (no deep_copy with length mistmatch)
    for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < NVAR_REG; i++) {
        nbody_data1(n, i) = nbody_data.h_view(n, i);
      } // end i
    } // end n
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

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus NBody::Fluxes
//! \brief Computes the "flux" in the nbody state, namely the rate of change in time for 
//! the mass, position and velocity for all bodies in the class. Acceleration on each body
//! is computed according to the pairwise gravitational attraction, and forcing by gas 
//! (gravity and accretion) which is populated during the HydroSrcTerms execution. All 
//! forces are computed on the same fine timestep, such that the NBody and Hydro state are 
//! evolved in exact tandem. 
TaskStatus NBody::Fluxes(Driver *pdrive, int stage) {

  // only perform this task on the primary MeshBlockPack on each rank
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // Steps 1 and 2 are only performed if sum_backreaction flagged
  if (sum_backreaction) {
    // Step 1: Iterate over all MeshBlockPacks on this rank (collect deltas and convert to rates)
    // Step 1a: Set backreaction register for this Mesh to zero
    Kokkos::deep_copy(delta_this_mesh, 0.0);
    for (int mbpid = 0; mbpid < pmy_pack->pmesh->nmb_packs_thisrank; mbpid++) {
      // Step 1a: Force comm from device to host for MeshBlockPack mbpid
      pmy_pack->pmesh->pmb_pack[mbpid].pnbody->delta_this_pack.template modify<DevExeSpace>();
      pmy_pack->pmesh->pmb_pack[mbpid].pnbody->delta_this_pack.template sync<HostMemSpace>();
      // Step 1b: Collect nbody deltas across MeshBlockPack mbpid
      Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
      for (int n = 0; n < num_nbody; n++) {
        for (int i = 0; i < NVAR_REG; i++) {
          if (i == 0) { // mdot = dm / (beta * dt)
            delta_this_mesh(n, i) += pmy_pack->pmesh->pmb_pack[mbpid].pnbody->delta_this_pack.h_view(n, i) / beta_dt;
          } else { // vdot = dp / (m * beta * dt)
            delta_this_mesh(n, i) += pmy_pack->pmesh->pmb_pack[mbpid].pnbody->delta_this_pack.h_view(n, i) / (pmy_pack->pmesh->pmb_pack[mbpid].pnbody->nbody_data.h_view(n, M_DATA) * beta_dt);
          }
        } // end register loop
      } // end body loop
      // Step 1c: Wipe register on MeshBlockPack mbpid and sync to device
      Kokkos::deep_copy(pmy_pack->pmesh->pmb_pack[mbpid].pnbody->delta_this_pack.view_host(), 0.0);
      pmy_pack->pmesh->pmb_pack[mbpid].pnbody->delta_this_pack.template modify<HostMemSpace>();
      pmy_pack->pmesh->pmb_pack[mbpid].pnbody->delta_this_pack.template sync<DevExeSpace>();
    } // end mbp loop

    // Step 2: Sum rates across all ranks (untested with MPI compile)
    Kokkos::deep_copy(delta_all_meshes, delta_this_mesh); // copy into comm buffer
    #if MPI_PARALLEL_ENABLED 
    if (global_variable::my_rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, delta_all_meshes.data(), NVAR_BACK, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    } else {
      MPI_Reduce(delta_all_meshes.data(), delta_all_meshes.data(), NVAR_BACK, MPI_ATHENA_REAL, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    #endif
  } // end if backreaction
  
  // Step 3: Skip following computations if not root processes
  // if (global_variable::my_rank != 0) return TaskStatus::complete;
  
  // Step 4: Set nbody flux to zero for entire register
  Kokkos::deep_copy(nbody_flux, 0.0);

  // Step 5: Compute flux for each body in class
  for (int n = 0; n < num_nbody; n++) {
    // Step 5a: Extract flux data (xdot = v, vdot = a)
    nbody_flux(n, MDOT_REG) = (inc_backreaction) ? delta_all_meshes(n, MDOT_BACK) : 0.0;

    // dot(x) = v
    nbody_flux(n, XDOT_REG) = nbody_data.h_view(n, VX_DATA);
    nbody_flux(n, YDOT_REG) = nbody_data.h_view(n, VY_DATA);
    nbody_flux(n, ZDOT_REG) = nbody_data.h_view(n, VZ_DATA);
    
    // dot(v) = dot(p) / m 
    nbody_flux(n, VXDOT_REG) = (inc_backreaction) ? delta_all_meshes(n, AX_GRAV_BACK) + delta_all_meshes(n, AX_ACC_BACK) : 0.0;
    nbody_flux(n, VYDOT_REG) = (inc_backreaction) ? delta_all_meshes(n, AY_GRAV_BACK) + delta_all_meshes(n, AY_ACC_BACK) : 0.0;
    nbody_flux(n, VZDOT_REG) = (inc_backreaction) ? delta_all_meshes(n, AZ_GRAV_BACK) + delta_all_meshes(n, AZ_ACC_BACK) : 0.0;

    // all later indices of nbody_flux left as ZERO (non-evolving registers)

    // Step 5b: add acceleraton by mutual nbody gravity
    for (int m = 0; m < num_nbody; m++) {
      if (m == n) continue; // no self-gravity
        
      // extract spatial seperation
      const Real dx = nbody_data.h_view(n, X_REG) - nbody_data.h_view(m, X_REG);
      const Real dy = nbody_data.h_view(n, Y_REG) - nbody_data.h_view(m, Y_REG);
      const Real dz = nbody_data.h_view(n, Z_REG) - nbody_data.h_view(m, Z_REG);
      Real r_sqr = SQR(dx) + SQR(dy) + SQR(dz);

      // branch if PostNewtonian forcing to be included
      if (!inc_pn) {
        // compute Newtnonian pairwise acceleration
        const Real g_fac = G_const * nbody_data.h_view(m, M_REG) / (r_sqr * std::sqrt(r_sqr));
        
        // decompose acceleration and update
        nbody_flux(n, VXDOT_REG) -= g_fac * dx;
        nbody_flux(n, VYDOT_REG) -= g_fac * dy;
        nbody_flux(n, VZDOT_REG) -= g_fac * dz; 
      } else {
        // include additional expansion terms up to 2.5PN (WIP)
        // formulaism from Eq 203 of "Gravitational Radiation from Post-Newtonian Sources 
        // and Inspiralling Compact Binaries" Blanchet 2014
        // notation switch from (1,2)->(n,m)

        // TODO: the form given by Blanchet is very bulky, consider other 
        // numeric implementations for faster forms

        // label masses
        const Real m1 = nbody_data.h_view(n, M_DATA);
        const Real m2 = nbody_data.h_view(m, M_DATA);
        const Real m1m2 = m1 * m2;
        const Real m2m2 = m2 * m2;
        const Real G2 = SQR(G_const);
        const Real G3 = G_const * G2;

        // extract velocity terms
        const Real dvx = nbody_data.h_view(n, VX_DATA) - nbody_data.h_view(m, VX_DATA);
        const Real dvy = nbody_data.h_view(n, VY_DATA) - nbody_data.h_view(m, VY_DATA);
        const Real dvz = nbody_data.h_view(n, VZ_DATA) - nbody_data.h_view(m, VZ_DATA);
        Real v_sqr = SQR(dvx) + SQR(dvy) + SQR(dvz);
        Real v1v2  = nbody_data.h_view(n, VX_DATA) * nbody_data.h_view(m, VX_DATA) + 
                     nbody_data.h_view(n, VY_DATA) * nbody_data.h_view(m, VY_DATA) +
                     nbody_data.h_view(n, VZ_DATA) * nbody_data.h_view(m, VZ_DATA);
        const Real v1_sqr = SQR(nbody_data.h_view(n, VX_DATA)) + nbody_data.h_view(n, VY_DATA) + nbody_data.h_view(n, VZ_DATA);
        const Real v2_sqr = SQR(nbody_data.h_view(m, VX_DATA)) + nbody_data.h_view(m, VY_DATA) + nbody_data.h_view(m, VZ_DATA);


        // compute unit directions
        const Real r = Kokkos::sqrt(r_sqr);
        const Real inv_r = 1.0 / r;
        const Real inv_r2 = SQR(inv_r);
        const Real inv_r3 = inv_r2 * inv_r;
        const Real n_x = dx * inv_r;
        const Real n_y = dy * inv_r;
        const Real n_z = dz * inv_r;

        // TODO: add unit conversions here, including for _G

        // 0th order (Newtonian)
        const Real newtonian_fac = - G_const * m2 * inv_r2;
        Real a_x = newtonian_fac * n_x;
        Real a_y = newtonian_fac * n_y;
        Real a_z = newtonian_fac * n_z;

        // 1st order (v/c)^2
        Real f_1st = G2 * inv_r3 * (5 * m1m2 + 4 * m2m2); //

        // convert back to code units and 
        nbody_flux(n, VXDOT_REG) += a_x / unit_A;
        nbody_flux(n, VYDOT_REG) += a_y / unit_A;
        nbody_flux(n, VZDOT_REG) += a_z / unit_A;
      } // end pn branch
    } // end m loop
  } // end n loop

  // Step 6: Skip later steps in sum_backreaction not flagged
  if (!sum_backreaction) {
    if (verbose) {
      std::cout << "Completed stage " << stage << " of NBody::Fluxes on MeshBlockPack " << pmy_pack->pmesh->nmb_packs_thisrank
                << " on rank " << global_variable::my_rank << std::endl; 
    }
    return TaskStatus::complete;
  } 

  // Step 7: Copy gravitational and accretion acceleration delta_all_meshes to nbody_data for storage, 
  // as delta_all_meshes wiped on next timestep. Copy only performed on last stage on integrator as no
  // history writes are performed on fine timestep
  if ((stage == (pdrive->nexp_stages))) { 
    for (int n = 0; n < num_nbody; n++) {
      nbody_data.h_view(n, MDOT_DATA)    = delta_all_meshes(n, MDOT_BACK);
      nbody_data.h_view(n, AX_GRAV_DATA) = delta_all_meshes(n, AX_GRAV_BACK);
      nbody_data.h_view(n, AY_GRAV_DATA) = delta_all_meshes(n, AY_GRAV_BACK);
      nbody_data.h_view(n, AZ_GRAV_DATA) = delta_all_meshes(n, AZ_GRAV_BACK);
      nbody_data.h_view(n, AX_ACC_DATA)  = delta_all_meshes(n, AX_ACC_BACK);
      nbody_data.h_view(n, AY_ACC_DATA)  = delta_all_meshes(n, AY_ACC_BACK);
      nbody_data.h_view(n, AZ_ACC_DATA)  = delta_all_meshes(n, AZ_ACC_BACK);
    } // end body loop
  } // end acceleration copy

  // Step 8: Report completion, if flagged
  if (verbose) {
    std::cout << "Completed stage " << stage << " of NBody::Fluxes on MeshBlockPack " << pmy_pack->pmesh->nmb_packs_thisrank
              << " on rank " << global_variable::my_rank << std::endl; 
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus NBody::RKUpdate
//! \brief Performs explicit update of the nbody state (nbody_data) for each stage of the
//! integration (set by the Hydro integrator), using weighted average and partial time step 
//! updates of the pairwise gravity and hydro forces.
TaskStatus NBody::RKUpdate(Driver *pdrive, int stage) {

  // Step 1: Perform integration only on root process, in principle MeshBlockPack
  // TEMP: run on all if (global_variable::my_rank != 0) return TaskStatus::complete;
  // if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // load integration weights from general time integrator
  Real &gam0 = pdrive->gam0[stage-1];
  Real &gam1 = pdrive->gam1[stage-1];
  Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);

  // Step 1: only perform integration on ONE mb_pack on ONE rank
  // if (global_variable::my_rank != 0) return TaskStatus::complete;
  // if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // Step 2: bump nbody register to next fine timestep
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < NVAR_REG; i++) { // skip iteration over "static" indices beyond NVAR_REG
          nbody_data.h_view(n, i) = gam0 * nbody_data.h_view(n, i) + gam1 * nbody_data1(n, i) + beta_dt * nbody_flux(n, i);
      } // end i
  } // end n

  // Step 4: Report, if flagged
  if (verbose) {
    std::cout << "Completed stage " << stage << " of NBody::RKUpdate on MeshBlockPack " << pmy_pack->pmesh->nmb_packs_thisrank
              << " on rank " << global_variable::my_rank << std::endl; 
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  TaskStatus NBody::RKUpdate
//! \brief After NBody::RKUpdate, the root meshblockpack contains the updated nbody_data 
//! state on the host ONLY. Communicate this updated state to all NBody instances and 
//! force update on device in prep for next (fine) time step.
TaskStatus NBody::Send(Driver *pdrive, int stage) {

  // TEMP
  return TaskStatus::complete;

  // Step 1: Only perform send-rcev on principle MeshBlockPack
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // Step 2: Send-recv updated nbody_data state to all ranks
  // TEMP build scratch array, if feasible offload to hpp
  HostArray2D<Real> nbody_comm_buffer;
  Kokkos::realloc(nbody_comm_buffer,  num_nbody, NVAR_DATA); // <- wasteful, offload
  Kokkos::deep_copy(nbody_comm_buffer, nbody_data.view_host());
  #if MPI_PARALLEL_ENABLED 
    MPI_Bcast(nbody_comm_buffer.data(), NVAR_DATA, MPI_ATHENA_REAL, 0, MPI_COMM_WORLD);
  #endif
  Kokkos::deep_copy(nbody_data.view_host(), nbody_comm_buffer);

  // Step 3: Copy nbody_data state from this MeshBlockPack to all others
  for (int mbpid = 0; mbpid < pmy_pack->pmesh->nmb_packs_thisrank; mbpid++) {
    // Step 3a: Copy nbody_data on host
    if (mbpid != 0) {
      Kokkos::deep_copy(pmy_pack->pmesh->pmb_pack[mbpid].pnbody->nbody_data.view_host(), nbody_data.view_host());
    }
    // Step 3b: Force update of nbody_data on device
    pmy_pack->pmesh->pmb_pack[mbpid].pnbody->nbody_data.template modify<HostMemSpace>();
    pmy_pack->pmesh->pmb_pack[mbpid].pnbody->nbody_data.template sync<DevExeSpace>();
  } // end mbp loop

  // Step 4: Report, if flagged
  if (verbose) {
    std::cout << "Completed stage " << stage << " of NBody::Send on MeshBlockPack " << pmy_pack->pmesh->nmb_packs_thisrank
              << " on rank " << global_variable::my_rank << std::endl; 
  }

  return TaskStatus::complete;
}

} // end namespace nbody