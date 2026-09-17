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
  // principle registers for wider access
  Kokkos::realloc(nbody_data, num_nbody, var_per_body);
  Kokkos::realloc(delta_nbody_data, num_nbody, var_per_body);
  // TODO: the following are ONLY accessed on the host, could change type
  Kokkos::realloc(_y_init, num_nbody, _reg_per_body);
  Kokkos::realloc(_y_sub, num_nbody, _reg_per_body);
  Kokkos::realloc(_y_ret, num_nbody, _reg_per_body);
  Kokkos::realloc(_k_sub, num_nbody, _reg_per_body);
  
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

  // set timestep properties
  eta_dt = pin->GetOrAddReal("nbody", "eta_dt", 0.01);
  dt_old = CalcTimeStep();
  dt_new = dt_old; 

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

// return max timestep for stable evolution from nbody state
Real NBody::CalcTimeStep() {

  Real hm4_max = std::numeric_limits<float>::min();

  for (int n = 0; n < num_nbody; n++) {
    // compute total accerelation due to hydro forces on BH n
    const Real a_hydro_sqr = 0.0;  // TEMP: set to zero
    // test pairwise interactions
    for (int m = 0; m < num_nbody; m++) {
      if (m == n) continue; // no self-interaction
      // compute binary properties
      const Real Gm_bin = _G * (nbody_data.h_view(n, M_DATA) + nbody_data.h_view(m, M_DATA));
      const Real dx = nbody_data.h_view(n, X_DATA) - nbody_data.h_view(m, X_DATA);
      const Real dy = nbody_data.h_view(n, Y_DATA) - nbody_data.h_view(m, Y_DATA);
      const Real dz = nbody_data.h_view(n, Z_DATA) - nbody_data.h_view(m, Z_DATA);
      const Real dr_sqr = dx * dx + dy * dy + dz * dz;
      const Real dvx = nbody_data.h_view(n, VX_DATA) - nbody_data.h_view(m, VX_DATA);
      const Real dvy = nbody_data.h_view(n, VY_DATA) - nbody_data.h_view(m, VY_DATA);
      const Real dvz = nbody_data.h_view(n, VZ_DATA) - nbody_data.h_view(m, VZ_DATA);
      const Real dv_sqr = dvx * dvx + dvy * dvy + dvz * dvz;
      // compute timescales
      const Real hm2_fb = dv_sqr / dr_sqr;                              // flyby time
      const Real hm4_ff = Gm_bin * Gm_bin / (dr_sqr * dr_sqr * dr_sqr); // freefall time
      const Real hm4_hydro = a_hydro_sqr / dr_sqr;                        // gas acceleration time
      // combine timescales
      const Real hm4 = hm2_fb * hm2_fb + hm4_ff + hm4_hydro;
      hm4_max = std::max(hm4_max, hm4);
    }
  }

  // return unscaled timestep 
  return std::pow(hm4_max, -0.25);
}

void NBody::EvaluateF(DualArray2D<Real> y, DualArray2D<Real> &f) {
  
  // evaluate forcing function f = ydot for nbody state
  // y = (m, x, y, z, vx, vy, vz ....)
  // f = (0, vx, vy, vz, ax, ay, az ...)

  for (int n = 0; n < num_nbody; n++) {
    // mdot = 0 
    f.h_view(n, M_DATA) = 0.0; 

    // dot(x) = v
    f.h_view(n, X_DATA) = y.h_view(n, VX_DATA);
    f.h_view(n, Y_DATA) = y.h_view(n, VY_DATA);
    f.h_view(n, Z_DATA) = y.h_view(n, VZ_DATA);
    
    // dot(v) = a TODO: add accelerations by gas (gravity, accretion etc.)
    f.h_view(n, AX_DATA) = 0.0;
    f.h_view(n, AY_DATA) = 0.0;
    f.h_view(n, AZ_DATA) = 0.0;
    // add acceleraton by mutual nbody gravity
    for (int m = 0; m < num_nbody; m++) {
      if (m == n) continue; // no self-gravity
        
      // extract spatial seperation
      const Real dx = y.h_view(n, X_DATA) - y.h_view(m, X_DATA);
      const Real dy = y.h_view(n, Y_DATA) - y.h_view(m, Y_DATA);
      const Real dz = y.h_view(n, Z_DATA) - y.h_view(m, Z_DATA);
    
      // compute acceleration
      const Real r_sqr = dx * dx + dy * dy + dz * dz;
      const Real g_fac = _G * y.h_view(m, M_DATA) / (r_sqr * std::sqrt(r_sqr));
      
      // decompose acceleration and update
      f.h_view(n, VX_DATA) -= g_fac * dx;
      f.h_view(n, VY_DATA) -= g_fac * dy;
      f.h_view(n, VZ_DATA) -= g_fac * dz;
    } // end m loop
  } // end n loop

  return;
}

void NBody::NBodySrcTerms(const Real beta_dt) {

  // currently only gravity implemented
  if (src_gravity) {
    NBodyGravitySrcTerm(beta_dt);
  }

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
      for (int n = 0; n < num_nbody; n++) {

        // TEMP: force gravity by cell-center
        // const Real rho = prim(m, IDN, k, j, i);
        // const Real g_fac = beta_dt * 

        // TEMP: set backreaction as zero
        Real dm_back = 0, dvx_back = 0, dvy_back = 0, dvz_back = 0;

        // stash backreaction registers, with care for race conditions
        Kokkos::atomic_add(&nbody_data.d_view(n, DM_BACK)) = dm_back;
        Kokkos::atomic_add(&nbody_data.d_view(n, DVX_BACK)) = dvx_back;
        Kokkos::atomic_add(&nbody_data.d_view(n, DVY_BACK)) = dvy_back;
        Kokkos::atomic_add(&nbody_data.d_view(n, DVZ_BACK)) = dvz_back;
      }
    }); // end par_for

  

  return;
}

} // end nbody namespace