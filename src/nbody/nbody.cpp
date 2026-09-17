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

NBody::NBody(MeshBlockPack *ppack, ParameterInput *pin) :
  nbody_data("nbody_data",1,1),
  delta_nbody_data("delta_nbody_data",1,1),
  _y_init("y_init",1,1),
  _y_sub("y_sub",1,1),
  _y_ret("y_ret",1,1),
  _k_sub("k_sub",1,1),
  pmy_pack(ppack) {

  // determine array dimensions from user input
  num_nbody = pin->GetOrAddInteger("nbody", "num_nbody", 0);
  var_per_body = pin->GetOrAddInteger("nbody", "var_per_nbody", 7);
  _reg_per_body = 7; // RK4 subregisters always len 7 (m, 3x, 3vx)

  // set physics modules
  src_gravity = pin->GetOrAddBoolean("nbody", "src_gravity", false);
  src_accretion = pin->GetOrAddBoolean("nbody", "src_accretion", false);

  // initialise array space on device
  if (num_nbody > 0) {
    // principle register for wider access
    Kokkos::realloc(nbody_data, num_nbody, var_per_body);
    Kokkos::realloc(delta_nbody_data, num_nbody, var_per_body);
    // TODO: the following are ONLY accessed on the host, could change type
    Kokkos::realloc(_y_init, num_nbody, _reg_per_body);
    Kokkos::realloc(_y_sub, num_nbody, _reg_per_body);
    Kokkos::realloc(_y_ret, num_nbody, _reg_per_body);
    Kokkos::realloc(_k_sub, num_nbody, _reg_per_body);
  }
  
  // load initial nbody state from user input
  for (int n = 0; n < num_nbody; n++) {
    std::string nbody_header = "nbody";
    // mass, position and velocity data MUST be passed
    nbody_data.h_view(n, 0) = pin->GetReal(nbody_header, "m" + std::to_string(n));
    nbody_data.h_view(n, 1) = pin->GetReal(nbody_header, "x" + std::to_string(n));
    nbody_data.h_view(n, 2) = pin->GetReal(nbody_header, "y" + std::to_string(n));
    nbody_data.h_view(n, 3) = pin->GetReal(nbody_header, "z" + std::to_string(n));
    nbody_data.h_view(n, 4) = pin->GetReal(nbody_header, "vx" + std::to_string(n));
    nbody_data.h_view(n, 5) = pin->GetReal(nbody_header, "vy" + std::to_string(n));
    nbody_data.h_view(n, 6) = pin->GetReal(nbody_header, "vz" + std::to_string(n));
    // all other reads optional, add overwrite
  } // end n

  // set delta_nbody as zero for first timestep
  // TODO: add restart protection? maybe not needed
  // Kokkos::deep_copy(delta_nbody_data, 0.0);

  // mark host view as modified and sync to device
  nbody_data.template modify<HostMemSpace>();
  delta_nbody_data.template modify<HostMemSpace>();

  nbody_data.template sync<DevExeSpace>();
  delta_nbody_data.template sync<DevExeSpace>();

} // end ctor

// nbody destructor: delete/free memory from internal objects
NBody::~NBody() {
}

void NBody::EvaluateF(DvceArray2D<Real> y, DvceArray2D<Real> &f) {
  
  // evaluate forcing function f = ydot for nbody state
  // y = (m, x, y, z, vx, vy, vz ....)
  // f = (0, vx, vy, vz, ax, ay, az ...)

  for (int n = 0; n < num_nbody; n++) {
    const Real offset_n = n * var_per_body;
    // mdot = 0 
    f.h_view(offset_n + 0) = 0.0; 

    // dot(x) = v
    f.h_view(offset_n + 1) = y.h_view(offset_n + 4);
    f.h_view(offset_n + 2) = y.h_view(offset_n + 5);
    f.h_view(offset_n + 3) = y.h_view(offset_n + 6);
    
    // dot(v) = a TODO: add accelerations by gas (gravity, accretion etc.)
    // TEMP: ax = 1.0
    f.h_view(offset_n + 4) = 1.0;
    f.h_view(offset_n + 5) = 0.0;
    f.h_view(offset_n + 6) = 0.0;
    // add acceleraton by mutual nbody gravity
    for (int m = 0; m < num_nbody; m++) {
      if (m == n) continue; // no self-gravity
      const Real offset_m = m * var_per_body;
        
      // extract spatial seperation
      const Real dx = y.h_view(offset_n + 1) - y.h_view(offset_m + 1);
      const Real dy = y.h_view(offset_n + 2) - y.h_view(offset_m + 2);
      const Real dz = y.h_view(offset_n + 3) - y.h_view(offset_m + 3);
    
      // compute acceleration
      const Real r_sqr = dx * dx + dy * dy + dz * dz;
      const Real a_fac = _G * y.h_view(offset_m) / (r_sqr * sqrt(r_sqr));
      
      // decompose acceleration and update
      f(offset_n + 4) -= a_fac * dx;
      f(offset_n + 5) -= a_fac * dy;
      f(offset_n + 6) -= a_fac * dz;
    } // end m loop
  } // end n loop

  return;
}

void NBody::NBodySrcTerms(const Real beta_dt) {

  NBodyGravitySrcTerm(beta_dt);

  return;
}

void NBody::NBodyGravitySrcTerm(const Real beta_dt) {

  // unpack mb_pack metadata
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &prim = pmy_pack->phydro->w0;
  auto &cons = pmy_pack->phydro->u0;

  par_for("nbody_gravity_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) 
    {
      // compute gravitational force on each body
      // for (int n = 0; n < num_nbody; n++) {
      //   continue;
      // }
    }); // end par_for
  return;
}

} // end nbody namespace