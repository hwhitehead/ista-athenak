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

  id.integrate = tl["after_timeintegrator"]->AddTask(&NBody::Gather, this, none);
  id.integrate = tl["after_timeintegrator"]->AddTask(&NBody::Integrate, this, none);
  id.integrate = tl["after_timeintegrator"]->AddTask(&NBody::Scatter, this, none);
  
  return;
}

// collect forcing by hydro on nbody state
TaskStatus NBody::Gather(Driver *pdrive, int stage) {

  return TaskStatus::complete;

}

// propogate nbody state forward in time using RK4
TaskStatus NBody::Integrate(Driver *pdrive, int stage) {

  // for now, pseduo-integrate position in time (x = t)

  const Real dt = pmy_pack->pmesh->dt;
  nbody_data.h_view(0, 1) += dt;
  nbody_data.template modify<HostMemSpace>();
  nbody_data.template sync<DevExeSpace>();

    // // only perform integration on rank 0
    // if (global_variable::my_rank != 0) return TaskStatus::complete;

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



// scatter nbody state from rank 0 to all ranks
// OR evolve each nbody seperate and avoid scatter
TaskStatus NBody::Scatter(Driver *pdrive, int stage) {

  return TaskStatus::complete;

}


} // end namespace nbody