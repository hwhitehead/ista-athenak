//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file hydro.cpp
//! \brief implementation of NBody class constructor and assorted other functions

#include <iostream>
#include <string>
#include <algorithm>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "diffusion/viscosity.hpp"
#include "diffusion/conduction.hpp"
#include "srcterms/srcterms.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"
#include "bvals/bvals.hpp"
#include "hydro/hydro.hpp"

#include "nbody/nbody.hpp"

namespace nbody {

NBody::Nbody(MeshBlockPack *ppack, ParameterInput *pin) :
  nbody_data("nbody_data",1,1),
  _y_init("y_init",1),
  _y_sub("y_sub",1),
  _y_ret("y_ret",1),
  _k_sub("k_sub",1),
  pmy_pack(ppack) {

  // determine array dimensions from user input
  num_nbody = pin->GetOrAddInteger("nbody", "num_nbody", 0);
  _ver_per_body = pin->GetOrAddInteger("nbody", "var_per_nbody", 7);

  // initialise array space on device
  if (num_nbody > 0) {
    // principle register for wider access
    Kokkos::realloc(nbody_data, num_nbody, _var_per_body);
    int len_sub_register = 7 * num_nbody;
    Kokkos::realloc(_y_init, len_sub_register);
    Kokkos::realloc(_y_sub, len_sub_register);
    Kokkos::realloc(_y_ret, len_sub_register);
    Kokkos::realloc(_k_sub, len_sub_register);
  }
  
  // load initial nbody state from user input
  for (int n = 0; n < num_nbody; n++) {
    std::string nbody_header = "nbody" + std::to_string(n);
    // mass, position and velocity data MUST be passed
    nbody_data(n, 0) = GetReal(nbody_header, "m");
    nbody_data(n, 1) = GetReal(nbody_header, "x");
    nbody_data(n, 2) = GetReal(nbody_header, "y");
    nbody_data(n, 3) = GetReal(nbody_header, "z");
    nbody_data(n, 4) = GetReal(nbody_header, "vx");
    nbody_data(n, 5) = GetReal(nbody_header, "vy");
    nbody_data(n, 6) = GetReal(nbody_header, "vz");
    // all other reads optional, add overwrite
  } // end n

}

NBody::~NBody() {
    return;
}

} // end nbody namespace