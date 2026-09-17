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
#include "coordinates/cell_locations.hpp"

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
  // both counters below deprecated to enum counters
  var_per_body = NVAR_DATA; // set by NBodyDataIndices
  _reg_per_body = NVAR_REG; // set by NBodyRegisterIndices

  // set physics modules
  src_gravity = pin->GetOrAddBoolean("nbody", "src_gravity", false);
  src_accretion = pin->GetOrAddBoolean("nbody", "src_accretion", false);

  // principle registers for wider access
  // nbody_data and delta_nbody_data are dual on host/device
  Kokkos::realloc(nbody_data, num_nbody, NVAR_DATA);
  Kokkos::realloc(delta_nbody_data, num_nbody, NVAR_BACK);
  // TODO: the following are ONLY accessed on the host, could change type
  Kokkos::realloc(_y_init, num_nbody, NVAR_REG);
  Kokkos::realloc(_y_sub, num_nbody, NVAR_REG);
  Kokkos::realloc(_y_ret, num_nbody, NVAR_REG);
  Kokkos::realloc(_k_sub, num_nbody, NVAR_REG);
  
  // load initial nbody state from user input
  for (int n = 0; n < num_nbody; n++) {
    std::string nbody_header = "nbody";
    std::string str_n = std::to_string(n);
    // mass, position and velocity data MUST be passed
    nbody_data.h_view(n, M_DATA) = pin->GetReal(nbody_header, "m" + str_n);
    nbody_data.h_view(n, X_DATA) = pin->GetReal(nbody_header, "x" + str_n);
    nbody_data.h_view(n, Y_DATA) = pin->GetReal(nbody_header, "y" + str_n);
    nbody_data.h_view(n, Z_DATA) = pin->GetReal(nbody_header, "z" + str_n);
    nbody_data.h_view(n, VX_DATA) = pin->GetReal(nbody_header, "vx" + str_n);
    nbody_data.h_view(n, VY_DATA) = pin->GetReal(nbody_header, "vy" + str_n);
    nbody_data.h_view(n, VZ_DATA) = pin->GetReal(nbody_header, "vz" + str_n);
    nbody_data.h_view(n, R_SOFT_DATA) = pin->GetReal(nbody_header, "r_soft" + str_n);
    
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
    f.h_view(n, MDOT_REG) = 0.0; 

    // dot(x) = v
    f.h_view(n, XDOT_REG) = y.h_view(n, VX_REG);
    f.h_view(n, YDOT_REG) = y.h_view(n, VY_REG);
    f.h_view(n, ZDOT_REG) = y.h_view(n, VZ_REG);
    
    // dot(v) = a TODO: add accelerations by gas (gravity, accretion etc.)
    f.h_view(n, VXDOT_REG) = 0.0;
    f.h_view(n, VYDOT_REG) = 0.0;
    f.h_view(n, VZDOT_REG) = 0.0;
    // add acceleraton by mutual nbody gravity
    for (int m = 0; m < num_nbody; m++) {
      if (m == n) continue; // no self-gravity
        
      // extract spatial seperation
      const Real dx = y.h_view(n, X_REG) - y.h_view(m, X_REG);
      const Real dy = y.h_view(n, Y_REG) - y.h_view(m, Y_REG);
      const Real dz = y.h_view(n, Z_REG) - y.h_view(m, Z_REG);
    
      // compute acceleration
      const Real r_sqr = dx * dx + dy * dy + dz * dz;
      const Real g_fac = _G * y.h_view(m, M_REG) / (r_sqr * std::sqrt(r_sqr));
      
      // decompose acceleration and update
      f.h_view(n, VXDOT_REG) -= g_fac * dx;
      f.h_view(n, VYDOT_REG) -= g_fac * dy;
      f.h_view(n, VZDOT_REG) -= g_fac * dz;
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
  auto &size  = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &prim = pmy_pack->phydro->w0;
  auto &cons = pmy_pack->phydro->u0;
  auto &nbody_read = nbody_data;
  auto &delta_write = delta_nbody_data;
  auto grav_const = _G;

  par_for("nbody_gravity_src", DevExeSpace(), 0, nmb1, 0, num_nbody, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int n, const int k, const int j, const int i) 
    {
      // identify cell position
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);

      // compute body-cell seperation
      const Real dx = x - nbody_read.d_view(n, X_DATA);
      const Real dy = y - nbody_read.d_view(n, Y_DATA);
      const Real dz = z - nbody_read.d_view(n, Z_DATA);
      const Real dr_sqr = dx * dx + dy * dy + dz * dz;
      const Real dr = Kokkos::sqrt(dr_sqr);

      // compute Newtonian gravitational acceleration
      const Real rho = prim(m, IDN, k, j, i);
      const Real g_fac = grav_const * nbody_read.d_view(n, M_DATA) * Kokkos::pow(dr_sqr + nbody_read.d_view(n, R_SOFT_DATA) * nbody_read.d_view(n, R_SOFT_DATA), -1.5);
      const Real dp_fac = g_fac * beta_dt * rho;
      const Real dpx = dp_fac * dx;
      const Real dpy = dp_fac * dy;
      const Real dpz = dp_fac * dz;
      const Real dE = dp_fac * (dx * prim(m, IVX, k, j, i)
                                + dy * prim(m, IVY, k, j, i)
                                + dz * prim(m, IVZ, k, j, j));

      // apply updates to cell's conserved quantities
      cons(m, IM1, k, j, i) += dpx;
      cons(m, IM2, k, j, i) += dpy;
      cons(m, IM3, k, j, i) += dpz;
      cons(m, IEN, k, j, i) += dE;

      // compute backreaction on body TEMP: set as zero
      Real dm_back = 0, dvx_back = 0, dvy_back = 0, dvz_back = 0;

      // stash backreaction registers, with care for race conditions
      Kokkos::atomic_add(&delta_write.d_view(n, DM_BACK), dm_back);
      Kokkos::atomic_add(&delta_write.d_view(n, DVX_BACK), dvx_back);
      Kokkos::atomic_add(&delta_write.d_view(n, DVY_BACK), dvy_back);
      Kokkos::atomic_add(&delta_write.d_view(n, DVZ_BACK), dvz_back);
      // wait until NBody::Gather task to sync back to host
    }); // end par_for

  

  return;
}

} // end nbody namespace