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

  id.reduce_mesh = tl["after_timeintegrator"]->AddTask(&NBody::ReduceParentMesh, this, none);
  id.reduce_meshes = tl["after_timeintegrator"]->AddTask(&NBody::ReduceAllMeshes, this, id.reduce_mesh);
  id.integrate = tl["after_timeintegrator"]->AddTask(&NBody::Integrate, this, id.reduce_meshes);
  id.scatter = tl["after_timeintegrator"]->AddTask(&NBody::Scatter, this, id.integrate);
  id.calc_dt = tl["after_timeintegrator"]->AddTask(&NBody::NewTimeStep, this, id.scatter);

  return;
}

// taskstatus wrapper to CalcTimeStep
TaskStatus NBody::NewTimeStep(Driver *pdrive, int stage) {
  
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
    pmy_pack->pmesh->pmy_pack[mbp_id].pnbody->delta_this_pack.template modify<DeviceMemSpace>(); // TODO: shift this to source term?
    pmy_pack->pmesh->pmy_pack[mbp_id].pnbody->delta_this_pack.template sync<HostMemSpace>();
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
    Kokkoss::deep_copy(pmy_pack->pmesh->pmb_pack[mbp_id].pnbody->delta_this_pack.view_host(), 0.0);
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