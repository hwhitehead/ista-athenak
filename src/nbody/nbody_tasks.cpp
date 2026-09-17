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

  id.gather = tl["after_timeintegrator"]->AddTask(&NBody::Gather, this, none);
  id.integrate = tl["after_timeintegrator"]->AddTask(&NBody::Integrate, this, id.gather);
  id.scatter = tl["after_timeintegrator"]->AddTask(&NBody::Scatter, this, id.integrate);
  id.calc_dt = tl["after_timeintegrator"]->AddTask(&NBody::NewTimeStep, this, id.integrate);

  return;
}
// taskstatus wrapper to CalcTimeStep
TaskStatus NBody::NewTimeStep(Driver *pdrive, int stage) {
  
  // symmetrise nbody timestep with previous value
  const Real dt_current = CalcTimeStep();
  const Real dt_sqr = dt_current * dt_current;
  dt_new = dt_sqr / dt_old;
  std::cout << "dt_new = " << dt_new << std::endl;
  return TaskStatus::complete;
}

// collect forcing by hydro on nbody state
TaskStatus NBody::Gather(Driver *pdrive, int stage) {

  return TaskStatus::complete;

}

// propogate nbody state forward in time using RK4
TaskStatus NBody::Integrate(Driver *pdrive, int stage) {

  return TaskStatus::complete;

  const Real dt = pmy_pack->pmesh->dt;
  const Real dt_over_6 = dt / 6.0;

  // for now, run on all ranks independently 
  // if (global_variable::my_rank != 0) return TaskStatus::complete;

  // package data into scratch registers
  // i runs over m,3x,3vx...
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < _reg_per_body; i++) {
          _y_init.h_view(n, i) = nbody_data.h_view(n, i);
      } // end i
  } // end n

  // compute k coefficients for RK4 substeps
  
  // step 1
  EvaluateF(_y_init, _k_sub);
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < _reg_per_body; i++) {
          _y_sub.h_view(n, i) = _y_init.h_view(n, i) + 0.5 * dt * _k_sub.h_view(n, i);
          _y_ret.h_view(n, i) = _y_init.h_view(n, i) + dt_over_6 * _k_sub.h_view(n, i);
      } // end i
  } // end n
  
  // step 2
  EvaluateF(_y_sub, _k_sub);
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < _reg_per_body; i++) {
          _y_sub.h_view(n, i) = _y_init.h_view(n, i) + 0.5 * dt * _k_sub.h_view(n, i);
          _y_ret.h_view(n, i) += 2.0 * dt_over_6 * _k_sub.h_view(n, i);
      } // end i
  } // end n

  // step 3
  EvaluateF(_y_sub, _k_sub);
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < _reg_per_body; i++) {
          _y_sub.h_view(n, i) = _y_init.h_view(n, i) + dt * _k_sub.h_view(n, i);
          _y_ret.h_view(n, i) += 2.0 * dt_over_6 * _k_sub.h_view(n, i);
      } // end i
  } // end n

  // step 4
  EvaluateF(_y_sub, _k_sub);
  for (int n = 0; n < num_nbody; n++) {
      for (int i = 0; i < _reg_per_body; i++) {
          _y_ret.h_view(n, i) += dt_over_6 * _k_sub.h_view(n, i);
      } // end i
  } // end n

  // update main register
  // no need to empty sub-step registers, all overwritten in next
  for (int n = 0; n < num_nbody; n++) {
    for (int i = 0; i < _reg_per_body; i++) {
        nbody_data.h_view(n, i) = _y_ret.h_view(n, i);
      } // end i
  }

  // force update of device state
  nbody_data.template modify<HostMemSpace>();
  delta_nbody_data.template sync<DevExeSpace>();

  return TaskStatus::complete;
}

// scatter nbody state from rank 0 to all ranks
// OR evolve each nbody seperate and avoid scatter
TaskStatus NBody::Scatter(Driver *pdrive, int stage) {

  return TaskStatus::complete;

}


} // end namespace nbody