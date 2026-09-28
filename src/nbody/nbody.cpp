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

//----------------------------------------------------------------------------------------
//! \fn  NBody::NBody
//! \brief Constructor: initialises data structures and parameters for the NBody class
NBody::NBody(MeshBlockPack *ppack, ParameterInput *pin) :
  nbody_data("nbody_data",1,1),
  nbody_data1("nbody_data1",1,1),
  nbody_flux("nbody_flux",1,1),
  delta_this_pack("delta_nbody_data",1,1),
  delta_this_mesh("delta_this_mesh",1,1),
  delta_all_meshes("delta_all_meshes",1,1),
  pmy_pack(ppack) {

  // determine number of bodies to read from input
  num_nbody = pin->GetOrAddInteger("nbody", "num_nbody", 0);

  // set physics module options using user input
  src_gravity       = pin->GetOrAddBoolean("nbody", "src_gravity", false);
  src_local_iso     = pin->GetOrAddBoolean("nbody", "src_local_iso", false);
  src_accretion     = pin->GetOrAddBoolean("nbody", "src_accretion", false);
  inc_backreaction  = pin->GetOrAddBoolean("nbody", "inc_backreaction", false);
  sum_backreaction  = pin->GetOrAddBoolean("nbody", "sum_backreaction", false);
  if (inc_backreaction) sum_backreaction = true; // enforce summation if flagged for live backreaction
  inc_pn = pin->GetOrAddBoolean("nbody", "inc_pn", false);

  // set disc state variables TODO: export to seperate class?
  Mach = pin->GetOrAddReal("problem","Mach", 1.0);
  inv_Mach_sqr = 1.0 / SQR(Mach);

  // import unit conversions (else all unity, used for PN terms WIP)
  G_const = pin->GetOrAddReal("nbody", "G_const", 1.0);
  unit_L = pin->GetOrAddReal("nbody", "unit_L", 1.0);
  unit_M = pin->GetOrAddReal("nbody", "unit_M", 1.0);
  unit_T = pin->GetOrAddReal("nbody", "unit_T", 1.0);
  unit_V = unit_L / unit_T;
  unit_A = unit_A / unit_T;

  // import verbose flag
  verbose = pin->GetOrAddBoolean("nbody", "verbose", false);

  // principle registers for wider access
  Kokkos::realloc(nbody_data,  num_nbody, NVAR_DATA);    
  Kokkos::realloc(nbody_data1, num_nbody, NVAR_REG);        
  Kokkos::realloc(nbody_flux,  num_nbody, NVAR_REG);        
  Kokkos::realloc(delta_this_pack,  num_nbody, NVAR_BACK);   
  Kokkos::realloc(delta_this_mesh,  num_nbody, NVAR_BACK);  
  Kokkos::realloc(delta_all_meshes, num_nbody, NVAR_BACK);
  
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
  Kokkos::deep_copy(delta_this_mesh, 0.0);
  Kokkos::deep_copy(delta_all_meshes, 0.0);

  // mark host view as modified and sync to device
  nbody_data.template modify<HostMemSpace>();
  delta_this_pack.template modify<HostMemSpace>();

  nbody_data.template sync<DevExeSpace>();
  delta_this_pack.template sync<DevExeSpace>();
} // end ctor

//----------------------------------------------------------------------------------------
//! \fn  void NBody::~NBody
//! \brief Destructor, frees any memory allocated during construction
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
      const Real Gm_bin = G_const * (nbody_data.h_view(n, M_DATA) + nbody_data.h_view(m, M_DATA));
      const Real dx = nbody_data.h_view(n, X_DATA) - nbody_data.h_view(m, X_DATA);
      const Real dy = nbody_data.h_view(n, Y_DATA) - nbody_data.h_view(m, Y_DATA);
      const Real dz = nbody_data.h_view(n, Z_DATA) - nbody_data.h_view(m, Z_DATA);
      const Real dr_sqr = dx * dx + dy * dy + dz * dz;
      const Real dvx = nbody_data.h_view(n, VX_DATA) - nbody_data.h_view(m, VX_DATA);
      const Real dvy = nbody_data.h_view(n, VY_DATA) - nbody_data.h_view(m, VY_DATA);
      const Real dvz = nbody_data.h_view(n, VZ_DATA) - nbody_data.h_view(m, VZ_DATA);
      const Real dv_sqr = dvx * dvx + dvy * dvy + dvz * dvz;
      // compute timescales
      const Real hm2_fb = dv_sqr / dr_sqr;                                // flyby time
      const Real hm4_ff = Gm_bin * Gm_bin / (dr_sqr * dr_sqr * dr_sqr);   // freefall time
      const Real hm4_hydro = a_hydro_sqr / dr_sqr;                        // gas acceleration time
      // combine timescales
      const Real hm4 = hm2_fb * hm2_fb + hm4_ff + hm4_hydro;
      hm4_max = std::max(hm4_max, hm4);
    }
  }

  // return unscaled timestep 
  return std::pow(hm4_max, -0.25);
}

void NBody::NBodySrcTerms(const Real beta_dt) {
  
  if (src_gravity) {
    NBodyPointSrcTerm(beta_dt);
  }
  if (src_local_iso && pmy_pack->phydro->peos->eos_data.is_ideal) {
    NBodyIsoSrcTerm(beta_dt);
  }

  return;
}

// source gravity and accretion for each body
void NBody::NBodyPointSrcTerm(const Real beta_dt) {

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

  // NBody properties (bypass implicit this)
  auto &nbody_data_ = nbody_data;
  int num_nbody_ = num_nbody;
  auto &delta_this_pack_ = delta_this_pack;
  auto G_const_ = G_const;
  bool is_ideal = pmy_pack->phydro->peos->eos_data.is_ideal;
  bool sum_backreaction_ = sum_backreaction;
  bool src_local_iso_ = src_local_iso;
  bool src_accretion_ = src_accretion;

  par_for("nbody_gravity_src", 
          DevExeSpace(), 
          0, nmb1, 
          ks, ke, 
          js, je, 
          is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // identify cell position
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // loop over bodies in NBody class
      for (int n = 0; n < num_nbody_; n++) {
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
        const Real g_fac = G_const_ * nbody_data_.d_view(n, M_DATA) * Kokkos::pow(dr_soft, -3.0);
        const Real dp_grav_fac = -g_fac * beta_dt * rho;
        const Real dpx_grav = dp_grav_fac * dx;
        const Real dpy_grav = dp_grav_fac * dy;
        const Real dpz_grav = dp_grav_fac * dz;

        // apply accretion, if in sink radius and flagged
        Real drho_acc = 0, dpx_acc = 0, dpy_acc = 0, dpz_acc = 0;
        if (src_accretion_) {
          const Real r_ratio = dr_true / nbody_data_.d_view(n, R_SOFT_DATA);
          if (r_ratio < 2) { 
            // compute mass loss rate
            Real sink_rate = 1e3 * Kokkos::exp(-Kokkos::pow(r_ratio, 4.0));
            sink_rate = Kokkos::min(sink_rate, 0.9 / beta_dt);
            // only accrete down to floor value to avoid blowup
            const Real drho_floor = 1e-6 - rho; // TODO: set at runtime with pin
            if (drho_floor < 0) { // rho > rho_floor
              const Real rhodot = - rho * sink_rate;
              drho_acc = rhodot * beta_dt;
              // ensure accretion only drives to sink value
              drho_acc = Kokkos::max(drho_acc, drho_floor);
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
              dpx_acc = drho_acc * vxstar;
              dpy_acc = drho_acc * vystar;
              dpz_acc = drho_acc * vzstar;
            } // end +ve rho_to_flor check
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
                        + dpz_grav * prim(mb_id, IVZ, k, j, i);
          cons(mb_id, IEN, k, j, i) += dE; 
        }

        // compute backreaction on body 
        if (sum_backreaction_) {
          // convert from densities 
          const Real cell_volume = size.d_view(mb_id).dx1 * size.d_view(mb_id).dx2 * size.d_view(mb_id).dx3;
          
          // gravity backreaction
          const Real dPx_grav = -dpx_grav * cell_volume;
          const Real dPy_grav = -dpy_grav * cell_volume;
          const Real dPz_grav = -dpz_grav * cell_volume;

          // accretion backreaction
          const Real dm_tot   = -drho_acc * cell_volume; 
          const Real dPx_acc  = -dpx_acc * cell_volume;
          const Real dPy_acc  = -dpy_acc * cell_volume;
          const Real dPz_acc  = -dpz_acc * cell_volume;

          // stash backreaction registers, with care for race conditions
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DM_BACK), dm_tot);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPX_GRAV_BACK), dPx_grav);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPY_GRAV_BACK), dPy_grav);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPZ_GRAV_BACK), dPz_grav);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPX_ACC_BACK), dPx_acc);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPY_ACC_BACK), dPy_acc);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPZ_ACC_BACK), dPz_acc);
        }
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
  //auto &general_data_ = general_data; // TODO track energy lost by cooling
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
      const Real E_last = cons(mb_id, IEN, k, j, i);
      const Real dE = (E_kin + E_int) - E_last; 
      cons(mb_id, IEN, k, j, i) += dE;
      // store energy loss
      // Kokkos::atomic_add(&general_data_.d_view(DE_GENERAL), dE); // TODO: implement collection of this
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
    sum_omega_sqr += G_const * nbody_data.d_view(n, M_DATA) * Kokkos::pow(dr_sqr, -3.0);
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