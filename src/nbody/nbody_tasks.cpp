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

  id.integrate = tl["before_timeintegrator"]->AddTask(&NBody::Integrate, this, none);

  id.integrate = tl["after_timeintegrator"]->AddTask(&NBody::Communicate, this, none);
  
  return;
}

// propogate nbody forward in time
TaskStatus NBody::Integrate(Driver *pdrive, int stage) {

    // // package data into compact form
    // for (int n = 0; n < num_nbody; n++) {
    //     const Real offset_n = 7 * n;
    //     for (int i = 0; i < 7; i++) {
    //         y_init(offset_n + i) = nbody_data[n](i);
    //     } // end i
    // } // end n

    // // compute k coefficients for RK4 substeps
    // const Real dt_over_6 = tstep / 6.0;
    // EvaluteF(y_init, k_sub);

    return TaskStatus::complete;
}

// propogate nbody forward in time
TaskStatus NBody::Communicate(Driver *pdrive, int stage) {

  return TaskStatus::complete;

}

void NBody::EvaluateF(DvceArray1D<Real> y, DvceArray1D<Real> &f) {
  // evaluate forcing function f = ydot for nbody state y = (mi, xi, ...)

//   for (int n = 0; n < num_nbody; n++) {
//     const Real offset_n = n * _var_per_body;
//     // mass data is not evolved using RK4, copy directly
    
//     // dot(x) = v
//     f(offset_n + 0) = y(offset_n + 4);
//     f(offset_n + 1) = y(offset_n + 5);
//     f(offset_n + 2) = y(offset_n + 6);
    
//     // dot(v) = a
//     // TODO: add accelerations by gas (gravity, accretion etc.)
//     f(offset_n + 3) = 0.0;
//     f(offset_n + 4) = 0.0;
//     f(offset_n + 5) = 0.0;
//     // add acceleraton by mutual nbody gravity
//     for (int m = 0; m < num_nbody; m++) {
//         if (m == n) continue; // no self-gravity
//         const Real offset_m = m * _var_per_nbody;

//     }
//   }
  return;
}

} // end namespace nbody