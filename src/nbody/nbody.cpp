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
  delta_this_pack("delta_nbody_data",1,1),
  delta_this_mesh("delta_this_mesh",1,1),
  delta_all_meshes("delta_all_meshes",1,1),
  _y_init("y_init",1,1),
  _y_sub("y_sub",1,1),
  _y_ret("y_ret",1,1),
  _k_sub("k_sub",1,1),
  pmy_pack(ppack) {

  // determine array dimensions from user input
  num_nbody = pin->GetOrAddInteger("nbody", "num_nbody", 0);

  // set physics modules
  src_gravity = pin->GetOrAddBoolean("nbody", "src_gravity", false);
  src_local_iso = pin->GetOrAddBoolean("nbody", "src_local_iso", false);
  src_accretion = pin->GetOrAddBoolean("nbody", "src_accretion", false);
  inc_backreaction = pin->GetOrAddBoolean("nbody", "inc_backreaction", false);
  inc_pn = pin->GetOrAddBoolean("nbody", "inc_pn", false);

  // set disc state variables
  Mach = pin->GetOrAddReal("problem","Mach", 1.0);
  inv_Mach_sqr = 1.0 / SQR(Mach);

  // import unit conversions (else all unity)
  unit_L = pin->GetOrAddReal("nbody", "unit_L", 1.0);
  unit_M = pin->GetOrAddReal("nbody", "unit_M", 1.0);
  unit_T = pin->GetOrAddReal("nbody", "unit_T", 1.0);
  unit_V = unit_L / unit_T;
  unit_A = unit_A / unit_T;

  // principle registers for wider access
  // nbody_data and delta_nbody_data are dual on host/device
  Kokkos::realloc(nbody_data, num_nbody, NVAR_DATA);
  Kokkos::realloc(delta_this_pack, num_nbody, NVAR_BACK);
  // TODO: the following are ONLY accessed on the host, could change type
  Kokkos::realloc(delta_this_mesh, num_nbody, NVAR_BACK);
  Kokkos::realloc(delta_all_meshes, num_nbody, NVAR_BACK);
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
    nbody_data.h_view(n, R_AMR_DATA) = pin->GetOrAddReal(nbody_header, "r_amr" + str_n, 0.0);
    nbody_data.h_view(n, N_AMR_DATA) = pin->GetOrAddReal(nbody_header, "n_amr" + str_n, 0.0); // currently unusable
    // all other reads optional, add overwrite
  } // end n

  // set timestep properties
  eta_dt = pin->GetOrAddReal("nbody", "eta_dt", 0.01);
  dt_old = CalcTimeStep();
  dt_new = dt_old; 

  // set all back reaction registers to zero
  Kokkos::deep_copy(delta_this_pack.view_host(), 0.0);
  Kokkos::deep_copy(delta_this_mesh.view_host(), 0.0);
  Kokkos::deep_copy(delta_all_meshes.view_host(), 0.0);

  // mark host view as modified and sync to device
  nbody_data.template modify<HostMemSpace>();
  delta_this_pack.template modify<HostMemSpace>();
  delta_this_mesh.template modify<HostMemSpace>();
  delta_all_meshes.template modify<HostMemSpace>();

  nbody_data.template sync<DevExeSpace>();
  delta_this_pack.template sync<DevExeSpace>();
  delta_this_mesh.template sync<DevExeSpace>();
  delta_all_meshes.template sync<DevExeSpace>();

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
    f.h_view(n, MDOT_REG) = (inc_backreaction) ? delta_all_meshes.h_view(n, MDOT_REG) : 0.0;

    // dot(x) = v
    f.h_view(n, XDOT_REG) = y.h_view(n, VX_REG);
    f.h_view(n, YDOT_REG) = y.h_view(n, VY_REG);
    f.h_view(n, ZDOT_REG) = y.h_view(n, VZ_REG);
    
    // dot(v) = dot(p) / m (use m from start of integration)
    f.h_view(n, VXDOT_REG) = (inc_backreaction) ? delta_all_meshes.h_view(n, DPX_BACK) / nbody_data.h_view(n, M_DATA) : 0.0;
    f.h_view(n, VYDOT_REG) = (inc_backreaction) ? delta_all_meshes.h_view(n, DPY_BACK) / nbody_data.h_view(n, M_DATA) : 0.0;
    f.h_view(n, VZDOT_REG) = (inc_backreaction) ? delta_all_meshes.h_view(n, DPZ_BACK) / nbody_data.h_view(n, M_DATA) : 0.0;

    // add acceleraton by mutual nbody gravity
    for (int m = 0; m < num_nbody; m++) {
      if (m == n) continue; // no self-gravity
        
      // extract spatial seperation
      const Real dx = y.h_view(n, X_REG) - y.h_view(m, X_REG);
      const Real dy = y.h_view(n, Y_REG) - y.h_view(m, Y_REG);
      const Real dz = y.h_view(n, Z_REG) - y.h_view(m, Z_REG);
      Real r_sqr = SQR(dx) + SQR(dy) + SQR(dz);

      // branch if PostNewtonian forcing to be included
      if (!inc_pn) {
        // compute Newtnonina pairwise acceleration
        const Real g_fac = _G * y.h_view(m, M_REG) / (r_sqr * std::sqrt(r_sqr));
        
        // decompose acceleration and update
        f.h_view(n, VXDOT_REG) -= g_fac * dx;
        f.h_view(n, VYDOT_REG) -= g_fac * dy;
        f.h_view(n, VZDOT_REG) -= g_fac * dz; 
      } else {
        // include additional expansion terms up to 2.5PN (WIP)
        // formulaism from Eq 203 of "Gravitational Radiation from Post-Newtonian Sources 
        // and Inspiralling Compact Binaries" Blanchet 2014
        // notation switch from (1,2)->(n,m)


        // label masses
        Real m1 = y.h_view(n, M_REG);
        Real m2 = y.h_view(m, M_REG);

        // extract velocity terms
        const Real dvx = y.h_view(n, VX_REG) - y.h_view(m, VX_REG);
        const Real dvy = y.h_view(n, VY_REG) - y.h_view(m, VY_REG);
        const Real dvz = y.h_view(n, VZ_REG) - y.h_view(m, VZ_REG);
        Real v_sqr = SQR(dvx) + SQR(dvy) + SQR(dvz);
        Real v_dot = y.h_view(n, VX_REG) * y.h_view(m, VX_REG) + 
                      y.h_view(n, VY_REG) * y.h_view(m, VY_REG) +
                      y.h_view(n, VZ_REG) * y.h_view(m, VZ_REG);

        // compute unit directions
        Real r = Kokkos::sqrt(r_sqr);
        Real inv_r = 1.0 / r;
        Real inv_r_sqr = SQR(inv_r);
        Real n_x = dx * inv_r;
        Real n_y = dy * inv_r;
        Real n_z = dz * inv_r;

        // TODO: add unit conversions here, including for _G

        // 0th order (Newtonian)
        const Real newtonian_fac = -_G * m2 * inv_r_sqr;
        Real a_x = newtonian_fac * n_x;
        Real a_y = newtonian_fac * n_y;
        Real a_z = newtonian_fac * n_z;

        // TODO: add higher order terms here

        // convert back to code units and 
        f.h_view(n, VXDOT_REG) += a_x / unit_A;
        f.h_view(n, VYDOT_REG) += a_y / unit_A;
        f.h_view(n, VZDOT_REG) += a_z / unit_A;
      } // end pn branch
    } // end m loop
  } // end n loop

  return;
}

void NBody::NBodySrcTerms(const Real beta_dt) {
  
  if (src_gravity) {
    NBodyGravitySrcTerm(beta_dt);
  }
  if (src_local_iso && pmy_pack->phydro->peos->eos_data.is_ideal) {
    NBodyIsoSrcTerm(beta_dt);
  }

  return;
}

// apply BH gravity to all cells
void NBody::NBodyGravitySrcTerm(const Real beta_dt) {

  // unpack all data pre par_for

  // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &prim = pmy_pack->phydro->w0;
  auto &cons = pmy_pack->phydro->u0;

  // TEMP report nmb_thispack
  std::cout << "There are " << pmy_pack->nmb_thispack << " meshblocks in this pack" << std::endl;
  std::cout << "There are " << num_nbody << " bodies in this pack" << std::endl;

  // NBody properties
  auto &nbody_data_ = nbody_data;
  auto &delta_this_pack_ = delta_this_pack;
  auto grav_const = _G;
  bool is_ideal = pmy_pack->phydro->peos->eos_data.is_ideal;
  bool inc_backreaction_ = inc_backreaction;
  bool src_local_iso_ = src_local_iso;

  par_for("nbody_gravity_src", DevExeSpace(), 0, nmb1, 0, num_nbody - 1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int n, const int k, const int j, const int i) 
    {
      // identify cell position and volume
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);
      const Real cell_volume = size.d_view(mb_id).dx1 * size.d_view(mb_id).dx2 * size.d_view(mb_id).dx3;

      // compute body-cell seperation
      const Real dx = x - nbody_data_.d_view(n, X_DATA);
      const Real dy = y - nbody_data_.d_view(n, Y_DATA);
      const Real dz = z - nbody_data_.d_view(n, Z_DATA);
      const Real dr_sqr = dx * dx + dy * dy + dz * dz;
      const Real dr_true = Kokkos::sqrt(dr_sqr);
      const Real dr_soft = Kokkos::sqrt(dr_sqr + SQR(nbody_data_.d_view(n, R_SOFT_DATA)));

      // compute Newtonian gravitational acceleration
      const Real rho = prim(mb_id, IDN, k, j, i);
      // g_fac = Gm/r^3 where r is softened by r_soft 
      const Real g_fac = grav_const * nbody_data_.d_view(n, M_DATA) * Kokkos::pow(dr_soft, -3.0);
      const Real dp_grav_fac = -g_fac * beta_dt * rho;
      const Real dpx_grav = dp_grav_fac * dx;
      const Real dpy_grav = dp_grav_fac * dy;
      const Real dpz_grav = dp_grav_fac * dz;

      // apply accretion, if in sink radius and flagged
      Real drho_acc = 0, dpx_acc = 0, dpy_acc = 0, dpz_acc = 0;
      if (src_accretion) {
        const Real r_ratio = dr / nbody_data_.d_view(n, R_SOFT_DATA);
        if (r_ratio < 2) { 
          // compute mass loss rate
          Real sink_rate = Kokkos::exp(-Kokkos::pow(r_ratio, 4.0));
          sink_rate = Kokkos::min(sink_rate, 0.9 / beta_dt);
          const Real rhodot = - rho * sink_rate;
          // apply torque free sink
          const Real inv_r = 1.0 / (dr_true + 1e-12); // small softening
          const Real rhatx = dx * inv_r;
          const Real rhaty = dy * inv_r;
          const Real rhatz = dz * inv_r;
          const Real vx_n = nbody_data_.d_view(n, VX_DATA);
          const Real vy_n = nbody_data_.d_view(n, VY_DATA);
          const Real vz_n = nbody_data_.d_view(n, VZ_DATA);
          const Real dvdotrhat = (prim(mb_id, IVX, k, j, i) - vx_n) * rhatx 
                                + (prim(mb_id, IVY, k, j, i) - vy_n) * rhaty 
                                + (prim(mb_id, IVZ, k, j, i) - vz_n) * rhatz;
          const Real vxstar    = dvdotrhat * rhatx + vx_n;
          const Real vystar    = dvdotrhat * rhaty + vy_n;
          const Real vzstar    = dvdotrhat * rhatz + vz_n;
          drho_acc = rhodot * dt;
          dpx_acc = drho_acc * vxstar;
          dpy_acc = drho_acc * vystar;
          dpz_acc = drho_acc * vzstar;
        } // end in sink
      } // end src_accretion

      // apply updates to cell's conserved quantities
      cons(mb_id, IDN, k, j, i) += drho_acc;
      cons(mb_id, IM1, k, j, i) += dpx_grav + dpx_acc;
      cons(mb_id, IM2, k, j, i) += dpy_grav + dpy_acc;
      cons(mb_id, IM3, k, j, i) += dpz_grav + dpz_acc;

      // TODO: this should be computed using acceleration and fluxes on cell FACES
      if (is_ideal && !src_local_iso_) { // only compute energy change if ideal AND not forced iso
        const Real dE = dpx_grav * prim(mb_id, IVX, k, j, i)
                      + dpy_grav * prim(mb_id, IVY, k, j, i)
                      + dpz_grav * prim(mb_id, IVZ, k, j, j);
        cons(mb_id, IEN, k, j, i) += dE; 
      }

      // compute backreaction on body 
      if (inc_backreaction_) {
        // gravity backreaction
        const Real dPx_grav = -dpx_grav * cell_volume;
        const Real dPy_grav = -dpy_grav * cell_volume;
        const Real dPz_grav = -dpz_grav * cell_volume;

        // accretion backreaction
        const Real dm_tot = -drho_acc * cell_volume; 
        const Real dPx_acc = -dpx_acc * cell_volume;
        const Real dPy_acc = -dpy_acc * cell_volume;
        const Real dPz_acc = -dpz_acc * cell_volume;

        const Real dPx_tot = dPx_grav + dPx_acc;
        const Real dPy_tot = dPy_grav + dPy_acc;
        const Real dPz_tot = dPz_grav + dPz_acc;

        // stash backreaction registers, with care for race conditions
        // TODO: stash these seperately for analysis and sum for integration
        Kokkos::atomic_add(&delta_this_pack_.d_view(n, DM_BACK), dm_tot);
        Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPX_BACK), dPx_tot);
        Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPY_BACK), dPy_tot);
        Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPZ_BACK), dPz_tot);
      }
    }); // end par_for
    
  return;
}

// update cell energy to match local isotherm
void NBody::NBodyIsoSrcTerm(const Real beta_dt) {

  // unpack all data pre par_for

  // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &prim = pmy_pack->phydro->w0;
  auto &cons = pmy_pack->phydro->u0;

  // NBody properties
  auto &nbody_data_ = nbody_data;
  int num_nbody_ = num_nbody;
  Real inv_Mach_sqr_ = inv_Mach_sqr;
  const Real inv_gm1 = 1.0 / (pmy_pack->phydro->peos->eos_data.gamma - 1.0);

  par_for("nbody_iso_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // identify cell position
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // identify local sound speed
      Real abs_phi_sum = 0.0;
      for (int n = 0; n < num_nbody_; n++) {
        const Real dx = x - nbody_data_.d_view(n, X_DATA);
        const Real dy = y - nbody_data_.d_view(n, Y_DATA);
        const Real dz = z - nbody_data_.d_view(n, Z_DATA);
        const Real dr_sqr = SQR(dx) + SQR(dy) + SQR(dz);
        const Real abs_phi_n = nbody_data_.d_view(n, M_DATA) * Kokkos::pow(dr_sqr, -0.5);
        abs_phi_sum += abs_phi_n;
      }
      const Real cs_sqr_local = abs_phi_sum * inv_Mach_sqr_;

      // compute local kinetic energy
      const Real rho = prim(mb_id, IDN, k, j, i);
      const Real E_kin = 0.5 * rho * (SQR(prim(mb_id, IVY, k, j, i)) + SQR(prim(mb_id, IVY, k, j, i)) + SQR(prim(mb_id, IVZ, k, j, i)));

      // assert local internal energy
      const Real E_int = cs_sqr_local * rho * inv_gm1;

      // set local energy
      cons(mb_id, IEN, k, j, i) = E_kin + E_int;
    }); // end par_for
    
  return;
}

// TODO integrate or deprecated the two funcs below

// compute sum of squared orbital frequencies
Real NBody::CalcLocalOmegaSqr(const Real x, const Real y, const Real z) {
  Real sum_omega_sqr = 0.0;

  for (int n = 0; n < num_nbody; n++) {
    const Real dx = x - nbody_data.d_view(n, X_DATA);
    const Real dy = y - nbody_data.d_view(n, Y_DATA);
    const Real dz = z - nbody_data.d_view(n, Z_DATA);
    const Real dr_sqr = SQR(dx) + SQR(dy) + SQR(dz);
    sum_omega_sqr += _G * nbody_data.d_view(n, M_DATA) * Kokkos::pow(dr_sqr, -3.0);
  }
  return sum_omega_sqr;
}

// compute local sound speed sqr using fixed Mach
Real CalcLocalSoundSpeedSqr(DualArray2D<Real> nbody_data, int num_nbody, Real inv_Mach_sqr, const Real x, const Real y, const Real z) {
  Real abs_phi_sum = 0;

  for (int n = 0; n < num_nbody; n++) {
    const Real dx = x - nbody_data.d_view(n, X_DATA);
    const Real dy = y - nbody_data.d_view(n, Y_DATA);
    const Real dz = z - nbody_data.d_view(n, Z_DATA);
    const Real dr_sqr = SQR(dx) + SQR(dy) + SQR(dz);
    const Real abs_phi_n = nbody_data.d_view(n, M_DATA) * Kokkos::pow(dr_sqr, -0.5);
    abs_phi_sum += abs_phi_n;
  }
  return abs_phi_sum * inv_Mach_sqr;
}

} // end nbody namespace