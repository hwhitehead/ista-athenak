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
#include "globals.hpp"

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
  nu_iso("nu_iso",1,1,1,1),
  max_nu_iso("max_nu_iso",1),
  pmy_pack(ppack) {

  // determine number of bodies to read from input
  num_nbody = pin->GetOrAddInteger("nbody", "num_nbody", 0);

  // set physics module options using user input
  src_gravity       = pin->GetOrAddBoolean("nbody", "src_gravity", false);
  src_local_iso     = pin->GetOrAddBoolean("nbody", "src_local_iso", false);
  src_blackbody     = pin->GetOrAddBoolean("nbody", "src_blackbody", false);
  src_accretion     = pin->GetOrAddBoolean("nbody", "src_accretion", false);
  inc_backreaction  = pin->GetOrAddBoolean("nbody", "inc_backreaction", false);
  sum_backreaction  = pin->GetOrAddBoolean("nbody", "sum_backreaction", false);
  if (inc_backreaction) sum_backreaction = true; // enforce summation if flagged for live backreaction
  inc_pn = pin->GetOrAddBoolean("nbody", "inc_pn", false);

  // check conflicting physics module options
  if (src_local_iso + src_blackbody + src_beta_cool > 1) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in NBody::NBody" << std::endl
                << "Multiple cooling source terms flagged in athinput" << std::endl
                << "Please select only one of src_local_iso, src_blackbody, or src_beta_cool" << std::endl;
    std::exit(EXIT_FAILURE);
    }
  }

  // set disc state variables
  Mach = pin->GetOrAddReal("nbody", "Mach", 1.0);
  inv_Mach_sqr = 1.0 / SQR(Mach);

  // import unit conversions (else all unity, used for PN terms WIP)
  // X_SI = X_CODE * unit_X
  G_const = pin->GetOrAddReal("nbody", "G_const", 1.0);
  unit_L  = pin->GetOrAddReal("nbody", "unit_L", 1.0);
  unit_M  = pin->GetOrAddReal("nbody", "unit_M", 1.0);
  unit_T  = pin->GetOrAddReal("nbody", "unit_T", 1.0);
  unit_V  = unit_L / unit_T;
  unit_A  = unit_A / unit_T;

  // opacity scaling (WIP, set as unity for now)
  kappa_es_code     = pin->GetOrAddReal("nbody", "kappa_es_code", 1.0);
  rad_const_code    = pin->GetOrAddReal("nbody", "rad_const_code", 1.0);
  c_light_code      = pin->GetOrAddReal("nbody", "c_light_code", 1.0);
  k_boltzmann_code  = pin->GetOrAddReal("nbody", "k_boltzmann_code", 1.0);
  mu_gas            = pin->GetOrAddReal("nbody", "mu_gas", 1.0);
  proton_mass_code  = pin->GetOrAddReal("nbody", "proton_mass_code", 1.0);
  gas_const_code    = k_boltzmann_code / (mu_gas * proton_mass_code);
  
  // import verbose flag
  verbose = pin->GetOrAddBoolean("nbody", "verbose", false);

  // determine viscosity usage
  alpha = pin->GetOrAddReal("nbody", "alpha", 0.0);
  if (alpha != 0.0) {
    // allocate array for viscosity coefficient, including ghosts
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int ncells1 = indcs.nx1 + 2 * (indcs.ng);
    int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
    int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;
    Kokkos::realloc(nu_iso, pmy_pack->nmb_thispack, ncells3, ncells2, ncells1); 
    // allocate register for tracking maximal viscosity
    Kokkos::realloc(max_nu_iso, pmy_pack->nmb_thispack);
  }

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
  
  // point mass src terms
  if (src_gravity) {
    NBodyPointSrcTerm(beta_dt);
  }

  // thermodynamic src terms
  if (pmy_pack->phydro->peos->eos_data.is_ideal) {
    if (src_local_iso) {
      NBodyForcedIsoSrcTerm(beta_dt);
    } else if (src_blackbody) {
      NBodyBlackBodySrcTerm(beta_dt);
    } else if (src_beta_cool) {
      NBodyBetaCoolSrcTerm(beta_dt);
    }
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
            Real sink_rate = 100 * Kokkos::exp(-Kokkos::pow(r_ratio, 4.0));
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
void NBody::NBodyForcedIsoSrcTerm(const Real beta_dt) {

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

  par_for("nbody_forced_iso_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // identify cell position
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // compute local sound speed as quadrature sum of Keplertian orbital velocities about each body
      Real sum_v_sqr = 0.0;
      for (int n = 0; n < num_nbody_; n++) {
        const Real dx = x - nbody_data_.d_view(n, X_DATA);
        const Real dy = y - nbody_data_.d_view(n, Y_DATA);
        const Real dz = z - nbody_data_.d_view(n, Z_DATA);
        const Real r_sqr = SQR(dx) + SQR(dy) + SQR(dz);
        const Real r = Kokkos::sqrt(r_sqr);
        const Real soft_r_sqr = r_sqr + SQR(nbody_data_.d_view(n, R_SOFT_DATA));
        const Real v_sqr = nbody_data_.d_view(n, M_DATA) * r / soft_r_sqr;
        sum_v_sqr += v_sqr;
      }
      const Real cs_sqr_local = sum_v_sqr * inv_Mach_sqr_;

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

// update cell energy according to black body cooling
void NBody::NBodyBlackBodySrcTerm(const Real beta_dt) {

  // function only valid for 2D systems, skip if 3D
  if (!pmy_pack->pmesh->three_d) {
    return;
  }

  // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &prim = pmy_pack->phydro->w0;
  auto &cons = pmy_pack->phydro->u0;
  const Real sqrt_3_over_4 = 0.25 * std::sqrt(3.0);

  // privatise nbody data for par_for
  const Real kappa_es_code_  = kappa_es_code;
  const Real gas_const_code_ = gas_const_code;
  const Real rad_const_code_ = rad_const_code;
  const Real c_light_code_   = c_light_code;

  par_for("nbody_blackbody_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // in 2D, all cells have fixed vertical size
      const Real two_H = size.h_view(mb_id).dx3;
      const Real one_over_4H = 0.5 / two_H;

      // compute cooling rate based on local density and temperature
      const Real tau = prim(mb_id, IDN, k, j, i) * two_H * kappa_es_code_;
      const Real tau_eff = 0.375 * tau + sqrt_3_over_4 + 0.25 / tau;
      const Real T_mid = prim(mb_id, IPR, k, j, i) / (prim(mb_id, IDN, k, j, i) * gas_const_code_);
      const Real T_eff = T_mid / tau_eff;
      const Real Q_cool = one_over_4H * rad_const_code_ * c_light_code_ * Kokkos::pow(T_eff, 4); 

      cons(mb_id, IEN, k, j, i) -= Q_cool * beta_dt;
    }); // end par_for
    
  return;
}

// update cell energy according to beta cooling
void NBody::NBodyBetaCoolSrcTerm(const Real beta_dt) {

  // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &prim = pmy_pack->phydro->w0;
  auto &cons = pmy_pack->phydro->u0;

  // privatise nbody data for par_for
  auto &nbody_data_ = nbody_data;
  int num_nbody_ = num_nbody;
  const Real inv_gm1 = 1.0 / (pmy_pack->phydro->peos->eos_data.gamma - 1.0);
  Real inv_Mach_sqr_ = inv_Mach_sqr;

  par_for("nbody_beta_cool_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {

      // identify cell position
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // compute target local sound speed as quadrature sum of Keplertian orbital velocities about each body
      Real sum_v_sqr = 0.0;
      Real sum_omega_sqr = 0.0;
      for (int n = 0; n < num_nbody_; n++) {
        const Real dx = x - nbody_data_.d_view(n, X_DATA);
        const Real dy = y - nbody_data_.d_view(n, Y_DATA);
        const Real dz = z - nbody_data_.d_view(n, Z_DATA);
        const Real r_sqr = SQR(dx) + SQR(dy) + SQR(dz);
        const Real r = Kokkos::sqrt(r_sqr);
        const Real soft_r_sqr = r_sqr + SQR(nbody_data_.d_view(n, R_SOFT_DATA));
        const Real v_sqr = nbody_data_.d_view(n, M_DATA) * r / soft_r_sqr;
        const Real omega_sqr = v_sqr / r_sqr;
        sum_v_sqr += v_sqr;
        sum_omega_sqr += omega_sqr;
      }
      const Real rho = prim(mb_id, IDN, k, j, i);
      const Real cs_sqr_target = sum_v_sqr * inv_Mach_sqr_ + 1e-8; // add floor value
      const Real cs_sqr_last = prim(mb_id, IPR, k, j, i) / rho;
      const Real t_cool = 2 * M_PI * Kokkos::pow(sum_omega_sqr, -0.5);
      const Real cs_sqr_next = (cs_sqr_last - cs_sqr_target) * Kokkos::exp(-beta_dt / t_cool) + cs_sqr_target;

      // assert local internal energy
      const Real E_int_last = cs_sqr_last * rho * inv_gm1;
      const Real E_int_next = cs_sqr_next * rho * inv_gm1;
      const Real dE_int = E_int_next - E_int_last;

      // set local energy
      cons(mb_id, IEN, k, j, i) += dE_int;
    }); // end par_for
    
  return;
}

// compute cell diffusivity according to beta cooling
void NBody::CalcViscousFluxAlpha(const DvceArray5D<Real> &w0) {

  // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  // par_for includes ghosts (dimension sensitive)
  int is = indcs.is - indcs.ng, ie = indcs.ie + indcs.ng;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  if (pmy_pack->pmesh->multi_d) {
    js -= indcs.ng; je += indcs.ng;
  }
  if (pmy_pack->pmesh->three_d) {
    ks -= indcs.ng; ke += indcs.ng;
  }
  int nmb1 = pmy_pack->nmb_thispack - 1;

  // set maximum register to zero
  Kokkos::deep_copy(max_nu_iso.view_host(), 0.0);
  max_nu_iso.template modify<HostMemSpace>();
  max_nu_iso.template sync<DevExeSpace>();
  const Real tiny_number = std::numeric_limits<double>::max();

  // privatise nbody data for par_for
  auto &nbody_data_ = nbody_data;
  auto &nu_iso_ = nu_iso;
  auto &max_nu_iso_ = max_nu_iso;
  int num_nbody_ = num_nbody;
  const Real alpha_ = alpha;
  Real inv_Mach_sqr_ = inv_Mach_sqr;
  bool src_local_iso_ = src_local_iso;

  par_for("nbody_calc_visc_flux", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {

      // identify cell position
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // compute Omega_tilde as quadrature sum of orbital frequency about each body
      Real omega_tilde_sqr = 0.0;
      Real sum_v_sqr = 0.0;
      for (int n = 0; n < num_nbody_; n++) {
        const Real dx = x - nbody_data_.d_view(n, X_DATA);
        const Real dy = y - nbody_data_.d_view(n, Y_DATA);
        const Real dz = z - nbody_data_.d_view(n, Z_DATA);
        const Real r_sqr = SQR(dx) + SQR(dy) + SQR(dz);
        const Real r = Kokkos::sqrt(r_sqr) + tiny_number;
        const Real soft_r_sqr = r_sqr + SQR(nbody_data_.d_view(n, R_SOFT_DATA));
        const Real v_sqr = nbody_data_.d_view(n, M_DATA) * r / soft_r_sqr;
        sum_v_sqr += v_sqr;
        const Real omega_sqr = nbody_data_.d_view(n, M_DATA) / (r * soft_r_sqr);
        omega_tilde_sqr += omega_sqr;
      }

      // calculate local kinematic viscosity
      Real cs_sqr;
      if (src_local_iso_) { // use assumed fixed Mach profile
        cs_sqr = sum_v_sqr * inv_Mach_sqr_;
      } else { // use local hydro state
        cs_sqr = w0(mb_id, IPR, k, j, i) / w0(mb_id, IDN, k, j, i);
      }
      Real nu_iso_local = alpha_ * cs_sqr * Kokkos::pow(omega_tilde_sqr, -0.5);
      // nu_iso_local = 5e-5;

      // stash viscosity state in register
      nu_iso_(mb_id, k, j, i) = nu_iso_local;

      // thread-safe maximum check for tracker
      Kokkos::atomic_max(&max_nu_iso_.d_view(mb_id), nu_iso_local);
    }); // end par_for
    
  // enforce update of maximum tracker on host
  max_nu_iso.template modify<DevExeSpace>();
  max_nu_iso.template sync<HostMemSpace>();

  // TEMP: verbose viscosity report
  for (int i = 0; i < pmy_pack->nmb_thispack; i++) {
    std::cout << "nu_iso_max (MeshBlock " << i << ") = " << max_nu_iso.h_view(i) << std::endl;
  }

  return;
}

// compute viscous fluxes according to inhomogeneous isotropic viscosity
void NBody::AddViscousFlux(const DvceArray5D<Real> &w0, const EOS_Data &eos,
    DvceFaceFld5D<Real> &flx) {

  // unpack mesh data per par_for
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto size = pmy_pack->pmb->mb_size;
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  auto &nu_iso_ = nu_iso;

  // fluxes in x1-direction
  int scr_level = 0;
  size_t scr_size = (ScrArray1D<Real>::shmem_size(ncells1)) * 3;
  auto flx1 = flx.x1f;

  par_for_outer("nbody_visc1", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // Add [2(dVx/dx)-(2/3)dVx/dx, dVy/dx, dVz/dx]
    par_for_inner(member, is, ie+1, [&](const int i) {
      fvx(i) = 4.0*(w0(m,IVX,k,j,i) - w0(m,IVX,k,j,i-1))/(3.0*size.d_view(m).dx1);
      fvy(i) =     (w0(m,IVY,k,j,i) - w0(m,IVY,k,j,i-1))/size.d_view(m).dx1;
      fvz(i) =     (w0(m,IVZ,k,j,i) - w0(m,IVZ,k,j,i-1))/size.d_view(m).dx1;
    });

    // In 2D/3D Add [(-2/3)dVy/dy, dVx/dy, 0]
    if (multi_d) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        fvx(i) -= ((w0(m,IVY,k,j+1,i) + w0(m,IVY,k,j+1,i-1)) -
                   (w0(m,IVY,k,j-1,i) + w0(m,IVY,k,j-1,i-1)))/(6.0*size.d_view(m).dx2);
        fvy(i) += ((w0(m,IVX,k,j+1,i) + w0(m,IVX,k,j+1,i-1)) -
                   (w0(m,IVX,k,j-1,i) + w0(m,IVX,k,j-1,i-1)))/(4.0*size.d_view(m).dx2);
      });
    }

    // In 3D Add [(-2/3)dVz/dz, 0,  dVx/dz]
    if (three_d) {
      par_for_inner(member, is, ie+1, [&](const int i) {
        fvx(i) -= ((w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k+1,j,i-1)) -
                   (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k-1,j,i-1)))/(6.0*size.d_view(m).dx3);
        fvz(i) += ((w0(m,IVX,k+1,j,i) + w0(m,IVX,k+1,j,i-1)) -
                   (w0(m,IVX,k-1,j,i) + w0(m,IVX,k-1,j,i-1)))/(4.0*size.d_view(m).dx3);
      });
    }

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie+1, [&](const int i) {
      Real nu1 = 0.5 * (nu_iso_(m,k,j,i) + nu_iso_(m,k,j,i-1));
      Real denf = 0.5 * (w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
      Real nud = nu1 * denf;
      //Real nud = 0.5 * (w0(m,IDN,k,j,i) * nu_iso_(m,k,j,i) + w0(m,IDN,k,j,i-1) * nu_iso_(m,k,j,i-1));
      //Real nud = 0.5*temp_static_nu_iso*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j,i-1));
      flx1(m,IVX,k,j,i) -= nud*fvx(i);
      flx1(m,IVY,k,j,i) -= nud*fvy(i);
      flx1(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        flx1(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k,j,i-1) + w0(m,IVX,k,j,i))*fvx(i) +
                                      (w0(m,IVY,k,j,i-1) + w0(m,IVY,k,j,i))*fvy(i) +
                                      (w0(m,IVZ,k,j,i-1) + w0(m,IVZ,k,j,i))*fvz(i));
      }
    });
  });
  if (pmy_pack->pmesh->one_d) {return;}

  // fluxes in x2-direction
  auto flx2 = flx.x2f;

  par_for_outer("nbody_visc2", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke, js, je+1,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // Add [(dVx/dy+dVy/dx), 2(dVy/dy)-(2/3)(dVx/dx+dVy/dy), dVz/dy]
    par_for_inner(member, is, ie, [&](const int i) {
      fvx(i) = (w0(m,IVX,k,j,i  ) - w0(m,IVX,k,j-1,i  ))/size.d_view(m).dx2 +
              ((w0(m,IVY,k,j,i+1) + w0(m,IVY,k,j-1,i+1)) -
               (w0(m,IVY,k,j,i-1) + w0(m,IVY,k,j-1,i-1)))/(4.0*size.d_view(m).dx1);
      fvy(i) = (w0(m,IVY,k,j,i) - w0(m,IVY,k,j-1,i))*4.0/(3.0*size.d_view(m).dx2) -
              ((w0(m,IVX,k,j,i+1) + w0(m,IVX,k,j-1,i+1)) -
               (w0(m,IVX,k,j,i-1) + w0(m,IVX,k,j-1,i-1)))/(6.0*size.d_view(m).dx1);
      fvz(i) = (w0(m,IVZ,k,j,i  ) - w0(m,IVZ,k,j-1,i  ))/size.d_view(m).dx2;
    });

    // In 3D Add [0, (-2/3)dVz/dz, dVy/dz]
    if (three_d) {
      par_for_inner(member, is, ie, [&](const int i) {
        fvy(i) -= ((w0(m,IVZ,k+1,j,i) + w0(m,IVZ,k+1,j-1,i)) -
                   (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k-1,j-1,i)))/(6.0*size.d_view(m).dx3);
        fvz(i) += ((w0(m,IVY,k+1,j,i) + w0(m,IVY,k+1,j-1,i)) -
                   (w0(m,IVY,k-1,j,i) + w0(m,IVY,k-1,j-1,i)))/(4.0*size.d_view(m).dx3);
      });
    }

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie, [&](const int i) {
      Real nu1 = 0.5 * (nu_iso_(m,k,j,i) + nu_iso_(m,k,j-1,i));
      Real denf = 0.5 * (w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
      Real nud = nu1 * denf;
      //Real nud = 0.5 * (w0(m,IDN,k,j,i) * nu_iso_(m,k,j,i) + w0(m,IDN,k,j-1,i) * nu_iso_(m,k,j-1,i));
      //Real nud = 0.5*temp_static_nu_iso*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
      flx2(m,IVX,k,j,i) -= nud*fvx(i);
      flx2(m,IVY,k,j,i) -= nud*fvy(i);
      flx2(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        flx2(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k,j-1,i) + w0(m,IVX,k,j,i))*fvx(i) +
                                      (w0(m,IVY,k,j-1,i) + w0(m,IVY,k,j,i))*fvy(i) +
                                      (w0(m,IVZ,k,j-1,i) + w0(m,IVZ,k,j,i))*fvz(i));
      }
    });
  });
  if (pmy_pack->pmesh->two_d) {return;}

  // fluxes in x3-direction
  auto flx3 = flx.x3f;

  par_for_outer("nbody_visc3", DevExeSpace(), scr_size, scr_level, 0, nmb1, ks, ke+1, js, je,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int k, const int j) {
    ScrArray1D<Real> fvx(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvy(member.team_scratch(scr_level), ncells1);
    ScrArray1D<Real> fvz(member.team_scratch(scr_level), ncells1);

    // Add [(dVx/dz+dVz/dx), (dVy/dz+dVz/dy), 2(dVz/dz)-(2/3)(dVx/dx+dVy/dy+dVz/dz)]
    par_for_inner(member, is, ie, [&](const int i) {
      fvx(i) = (w0(m,IVX,k,j,i  ) - w0(m,IVX,k-1,j,i  ))/size.d_view(m).dx3 +
              ((w0(m,IVZ,k,j,i+1) + w0(m,IVZ,k-1,j,i+1)) -
               (w0(m,IVZ,k,j,i-1) + w0(m,IVZ,k-1,j,i-1)))/(4.0*size.d_view(m).dx1);
      fvy(i) = (w0(m,IVY,k,j,i  ) - w0(m,IVY,k-1,j,i  ))/size.d_view(m).dx3 +
              ((w0(m,IVZ,k,j+1,i) + w0(m,IVZ,k-1,j+1,i)) -
               (w0(m,IVZ,k,j-1,i) + w0(m,IVZ,k-1,j-1,i)))/(4.0*size.d_view(m).dx2);
      fvz(i) = (w0(m,IVZ,k,j,i) - w0(m,IVZ,k-1,j,i))*4.0/(3.0*size.d_view(m).dx3) -
              ((w0(m,IVX,k,j,i+1) + w0(m,IVX,k-1,j,i+1)) -
               (w0(m,IVX,k,j,i-1) + w0(m,IVX,k-1,j,i-1)))/(6.0*size.d_view(m).dx1) -
              ((w0(m,IVY,k,j+1,i) + w0(m,IVY,k-1,j+1,i)) -
               (w0(m,IVY,k,j-1,i) + w0(m,IVY,k-1,j-1,i)))/(6.0*size.d_view(m).dx2);
    });

    // Sum viscous fluxes into fluxes of conserved variables; including energy fluxes
    par_for_inner(member, is, ie, [&](const int i) {
      Real nu1 = 0.5 * (nu_iso_(m,k,j,i) + nu_iso_(m,k-1,j,i));
      Real denf = 0.5 * (w0(m,IDN,k,j,i) + w0(m,IDN,k-1,j,i));
      Real nud = nu1 * denf;
      // Real nud = 0.5 * (w0(m,IDN,k,j,i) * nu_iso_(m,k,j,i) + w0(m,IDN,k-1,j,i) * nu_iso_(m,k-1,j,i));
      //Real nud = 0.5*temp_static_nu_iso*(w0(m,IDN,k,j,i) + w0(m,IDN,k,j-1,i));
      flx3(m,IVX,k,j,i) -= nud*fvx(i);
      flx3(m,IVY,k,j,i) -= nud*fvy(i);
      flx3(m,IVZ,k,j,i) -= nud*fvz(i);
      if (eos.is_ideal) {
        flx3(m,IEN,k,j,i) -= 0.5*nud*((w0(m,IVX,k-1,j,i) + w0(m,IVX,k,j,i))*fvx(i) +
                                      (w0(m,IVY,k-1,j,i) + w0(m,IVY,k,j,i))*fvy(i) +
                                      (w0(m,IVZ,k-1,j,i) + w0(m,IVZ,k,j,i))*fvz(i));
      }
    });
  });

  return;
}

void NBody::NewViscousTimeStep(const DvceArray5D<Real> &w0, const EOS_Data &eos_data) {
  // viscous timestep on MeshBlock(s) in this pack for inhomogeneous isotropic viscosity
  dt_visc = std::numeric_limits<float>::max();
  auto size = pmy_pack->pmb->mb_size;
  for (int m=0; m<(pmy_pack->nmb_thispack); ++m) {
    Real inv_dx2_sum = 1.0/SQR(size.h_view(m).dx1);
    Real inv_dx2_max = inv_dx2_sum;
    if (pmy_pack->pmesh->multi_d) {
      Real inv_dx2 = 1.0/SQR(size.h_view(m).dx2);
      inv_dx2_sum += inv_dx2;
      inv_dx2_max = std::max(inv_dx2_max, inv_dx2);
    }
    if (pmy_pack->pmesh->three_d) {
      Real inv_dx2 = 1.0/SQR(size.h_view(m).dx3);
      inv_dx2_sum += inv_dx2;
      inv_dx2_max = std::max(inv_dx2_max, inv_dx2);
    }
    // use maximum reported viscosity in this MeshBlock to compute rate
    Real rate = 2.0 * max_nu_iso.h_view(m) * (inv_dx2_sum + inv_dx2_max/3.0);
    dt_visc = pmy_pack->pmesh->cfl_no * std::min(dt_visc, 1.0/rate);
  }
  return;
}

// write NBody data to hst output TODO: internalise as standard output
void NBodyHistory(HistoryData *pdata, Mesh *pm) {
  // max number of history variables must be < NHISTORY_VARIABLES < NREDUCTION_VARIABLES
  // set in outputs.hpp and athena.hpp respectively 

  // by default, HistoryOuptut reduces across hist_data all ranks
  if (global_variable::my_rank != 0) return;

  // do not attempt write if no NBody instance
  if (pm->pmb_pack[0].pnbody == nullptr) return; 

  // generate labels for nbody data using first pack on this rank
  int num_nbody = pm->pmb_pack[0].pnbody->num_nbody;
  pdata->nhist = num_nbody * NVAR_HIST; 
  for (int n = 0; n < num_nbody; ++n) {
    int hist_offset = n * NVAR_HIST;
    pdata->label[M_HIST + hist_offset] = "m" + std::to_string(n); 
    pdata->label[X_HIST + hist_offset] = "x" + std::to_string(n);
    pdata->label[Y_HIST + hist_offset] = "y" + std::to_string(n);
    pdata->label[Z_HIST + hist_offset] = "z" + std::to_string(n);
    pdata->label[VX_HIST + hist_offset] = "vx" + std::to_string(n);
    pdata->label[VY_HIST + hist_offset] = "vy" + std::to_string(n);
    pdata->label[VZ_HIST + hist_offset] = "vz" + std::to_string(n);
    pdata->label[AX_GRAV_HIST + hist_offset] = "ax_grav" + std::to_string(n);
    pdata->label[AY_GRAV_HIST + hist_offset] = "ay_grav" + std::to_string(n);
    pdata->label[AZ_GRAV_HIST + hist_offset] = "az_grav" + std::to_string(n);
    pdata->label[AX_ACC_HIST + hist_offset] = "ax_acc" + std::to_string(n);
    pdata->label[AY_ACC_HIST + hist_offset] = "ay_acc" + std::to_string(n);
    pdata->label[AZ_ACC_HIST + hist_offset] = "az_acc" + std::to_string(n);
  } // end body loop

  // stash values
  int column_index = 0;
  for (int n = 0; n < num_nbody; ++n) {
    // write per-body output using nbody_data
    for (int i = M_HIST; i <= AZ_ACC_HIST; i++) {
      pdata->hdata[column_index] = pm->pmb_pack[0].pnbody->nbody_data.h_view(n, i);
      column_index++;
    } // end variable loop
  } // end body loop
  return;
}

// apply AMR to region about each body with non-zero refinement radius
void NBodyTrackRefinementCondition(MeshBlockPack* pmbp) {
  
  // do not attempt write if no NBody instance
  if (pmbp->pnbody == nullptr) return; 

  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &multi_d = pmbp->pmesh->multi_d;
  auto &three_d = pmbp->pmesh->three_d;

  // loop over MeshBlocks in this MeshBlockPack
  // MeshBlock count small, perfom on host
  for (int mb_id = 0; mb_id < nmb; mb_id++) {

    // by default, mark for derefine
    bool refine = false;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(mb_id).x1min;
    Real &x1max = size.h_view(mb_id).x1max;
    Real &x2min = size.h_view(mb_id).x2min;
    Real &x2max = size.h_view(mb_id).x2max;
    Real &x3min = size.h_view(mb_id).x3min;
    Real &x3max = size.h_view(mb_id).x3max;

    // cycle over bodies
    for (int n = 0; n < pmbp->pnbody->num_nbody; n++) {
      // if refinement radius for body is zero, skip
      const Real rad = pmbp->pnbody->nbody_data.h_view(n, R_AMR_DATA);
      if (rad == 0.0) continue;

      // save position 
      const Real x1 = pmbp->pnbody->nbody_data.h_view(n, X_DATA);
      const Real x2 = pmbp->pnbody->nbody_data.h_view(n, Y_DATA);
      const Real x3 = pmbp->pnbody->nbody_data.h_view(n, Z_DATA);
      
      // check overlap with AMR region and MeshBlock
      if (((x1min < (x1+rad)) && (x1min > (x1-rad))) ||
        ((x1max < (x1+rad)) && (x1max > (x1-rad))) ||
        ((x1max > (x1+rad)) && (x1min < (x1-rad)))) {
        if (!(multi_d) ||
          (((x2min < (x2+rad)) && (x2min > (x2-rad))) ||
          ((x2max < (x2+rad)) && (x2max > (x2-rad))) ||
          ((x2max > (x2+rad)) && (x2min < (x2-rad)))) ) {
          if (!(three_d) ||
            (((x3min < (x3+rad)) && (x3min > (x3-rad))) ||
            ((x3max < (x3+rad)) && (x3max > (x3-rad))) ||
            ((x3max > (x3+rad)) && (x3min < (x3-rad)))) ) {
            refine = true;
          }
        }
      }
    } // end n loop
    if (refine) {
      refine_flag.h_view(mb_id + mbs) = 1;
    } else {
      refine_flag.h_view(mb_id + mbs) = -1;
    }
  } // end mb_id loop

  // sync host and device  DevExeSpace
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

} // end nbody namespace