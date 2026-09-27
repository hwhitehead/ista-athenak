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
  id.initrk   = tl["stagen"]->AddTask(&NBody::InitRK, this, none);
  id.flux     = tl["stagen"]->AddTask(&NBody::Fluxes, this, id.initrk);
  id.rkupdt   = tl["stagen"]->AddTask(&NBody::RKUpdate, this, id.flux);
  id.newdt    = tl["stagen"]->AddTask(&NBody::NewTimeStep, this, id.rkupdt);

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

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::InitRK
//! \brief Simple task list function that copies nbody_data --> nbody_data1 in first 
//! stage. Extended to handle RK register logic at given stage
TaskStatus NBody::InitRK(Driver *pdrive, int stage) {
  
  // only run this task on the principle meshblock pack (rank0, mbpid=0)
  if (global_variable::my_rank != 0) return TaskStatus::complete;
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

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

  // evaluate time evoluton of the NBody state nbody_flux
  // only run this task on the principle meshblock pack (rank0, mbpid=0)
  // TODO: when MPI comm step included, ensure dev2host executed on all ranks
  if (global_variable::my_rank != 0) return TaskStatus::complete;
  if (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) return TaskStatus::complete;

  // Step 1: Force comm from device to host for backreaction registers
  delta_this_pack.template modify<DevExeSpace>();
  delta_this_pack.template sync<HostMemSpace>();

  // Step 2: Collect nbody deltas across system (WARNING: currently safe only for single MeshBlockPack)
  // Convert deltas into rates (e.g. dm into mdot), averaging over fine timestep
  // TODO: add MPI comm step to collect over ranks here
  if (sum_backreaction) {
    Real beta_dt = (pdrive->beta[stage-1])*(pmy_pack->pmesh->dt);
    for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < NVAR_REG; i++) {
        if (i == 0) { // mdot = dm / (beta * dt)
          delta_all_meshes(n, i) = delta_this_pack.h_view(n, i) / beta_dt;
        } else { // vdot = dp / (m * beta * dt)
          delta_all_meshes(n, i) = delta_this_pack.h_view(n, i) / (nbody_data.h_view(n, M_DATA) * beta_dt);
        }
      } // end register loop
    } // end body loop
  } // end if sum_backreaction
  
  // Step 3: Set nbody flux to zero for entire register
  Kokkos::deep_copy(nbody_flux, 0.0);

  // Step 4: Compute flux for each body in class
  for (int n = 0; n < num_nbody; n++) {
    // register now contains mdot, not delta m
    nbody_flux(n, MDOT_REG) = (inc_backreaction) ? delta_all_meshes(n, MDOT_BACK) : 0.0;

    // dot(x) = v
    nbody_flux(n, XDOT_REG) = nbody_data.h_view(n, VX_DATA);
    nbody_flux(n, YDOT_REG) = nbody_data.h_view(n, VY_DATA);
    nbody_flux(n, ZDOT_REG) = nbody_data.h_view(n, VZ_DATA);
    
    // dot(v) = dot(p) / m (use m from start of integration)
    // registers now contain accelerations, not momentum changes
    nbody_flux(n, VXDOT_REG) = (inc_backreaction) ? delta_all_meshes(n, AX_GRAV_BACK) + delta_all_meshes(n, AX_ACC_BACK) : 0.0;
    nbody_flux(n, VYDOT_REG) = (inc_backreaction) ? delta_all_meshes(n, AY_GRAV_BACK) + delta_all_meshes(n, AY_ACC_BACK) : 0.0;
    nbody_flux(n, VZDOT_REG) = (inc_backreaction) ? delta_all_meshes(n, AZ_GRAV_BACK) + delta_all_meshes(n, AZ_ACC_BACK) : 0.0;

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
        const Real newtonian_fac = - G_const * m2 * inv_r_sqr;
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

  // Step 5: Skip later steps in sum_backreaction not flagged
  if (!sum_backreaction) {
    // Step 6: Report, if flagged
    if (verbose) {
      std::cout << "Completed stage " << stage << " of NBody::Fluxes on MeshBlockPack " << pmy_pack->pmesh->nmb_packs_thisrank
                << " on rank " << global_variable::my_rank << std::endl; 
    }
    return TaskStatus::complete;
  } 

  // Step 6: Copy gravitational and accretion acceleration delta to data for storage
  // Copy only performed on last stage on integrator
  if ((stage == (pdrive->nexp_stages))) { 
    for (int n = 0; n < num_nbody; n++) {
      nbody_data.h_view(n, AX_GRAV_DATA) = delta_all_meshes(n, AX_GRAV_BACK);
      nbody_data.h_view(n, AY_GRAV_DATA) = delta_all_meshes(n, AY_GRAV_BACK);
      nbody_data.h_view(n, AZ_GRAV_DATA) = delta_all_meshes(n, AZ_GRAV_BACK);
      nbody_data.h_view(n, AX_ACC_DATA)  = delta_all_meshes(n, AX_ACC_BACK);
      nbody_data.h_view(n, AY_ACC_DATA)  = delta_all_meshes(n, AY_ACC_BACK);
      nbody_data.h_view(n, AZ_ACC_DATA)  = delta_all_meshes(n, AZ_ACC_BACK);
    } // end body loop
  } // end acceleration copy
  
  // Step 7: Cleanup collection registers for next finetimestep
  Kokkos::deep_copy(delta_all_meshes, 0.0);
  Kokkos::deep_copy(delta_this_mesh, 0.0); // TODO: currently unused as no MPI comm
  Kokkos::deep_copy(delta_this_pack.view_host(), 0.0);

  // Step 8: Enforce update of register state on device for DualArray delta_this_pack
  delta_this_pack.template modify<HostMemSpace>();
  delta_this_pack.template sync<DevExeSpace>();

  // Step 9: Report, if flagged
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
          nbody_data.h_view(n, i) = gam0 * nbody_data.h_view(n, i) + gam1 * nbody_data1(n, i) + beta_dt * nbody_flux(n, i);
      } // end i
  } // end n

  // Step 3: enforce update of nbody state on device for DualArray registers
  nbody_data.template modify<HostMemSpace>();
  nbody_data.template sync<DevExeSpace>();

  // Step 4: Report, if flagged
  if (verbose) {
    std::cout << "Completed stage " << stage << " of NBody::RKUpdate on MeshBlockPack " << pmy_pack->pmesh->nmb_packs_thisrank
              << " on rank " << global_variable::my_rank << std::endl; 
  }

  return TaskStatus::complete;
}

} // end namespace nbody