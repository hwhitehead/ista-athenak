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
#include "mhd/mhd.hpp"
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
  nbody_comm("nbody_comm",1,1),
  delta_this_pack("delta_nbody_data",1,1),
  delta_this_mesh("delta_this_mesh",1,1),
  delta_all_meshes("delta_all_meshes",1,1),
  phi("phi",1,1,1,1,1),
  gv("gv",1,1,1,1,1),
  gf("gf",1,1,1,1,1),
  nu_iso("nu_iso",1,1,1,1),
  max_nu_iso("max_nu_iso",1),
  pmy_pack(ppack) {

  // determine number of bodies to read from input
  num_nbody = pin->GetOrAddInteger("nbody", "num_nbody", 0);

  // set physics module options using user input
  src_gravity       = pin->GetOrAddBoolean("nbody", "src_gravity", false);
  src_accretion     = pin->GetOrAddBoolean("nbody", "src_accretion", false);
  cooling_mode      = pin->GetOrAddInteger("nbody", "cooling_mode", 0); // typed by NBodyCoolingModes in nbody.hpp

  // set back reaction options
  inc_backreaction  = pin->GetOrAddBoolean("nbody", "inc_backreaction", false);
  sum_backreaction  = pin->GetOrAddBoolean("nbody", "sum_backreaction", false);
  if (inc_backreaction) sum_backreaction = true; // enforce summation if flagged for live backreaction
  inc_pn = pin->GetOrAddBoolean("nbody", "inc_pn", false); // WIP, inactive

  // set disc state variables
  Mach           = pin->GetOrAddReal("nbody", "Mach", 1.0);
  inv_Mach_sqr   = 1.0 / SQR(Mach);
  rho_sink_floor = pin->GetOrAddReal("nbody", "rho_sink_floor", 1e-8);
  cs_sqr_floor   = pin->GetOrAddReal("nbody", "cs_sqr_floor", 1e-6);
  
  // set sink properties
  sink_mode      = pin->GetOrAddInteger("nbody", "sink_mode", 0); // 0 = all, 1 = flat, 2 = adapt nu, 3 = ideal nu, 4 = adapt cs
  soft_mode      = pin->GetOrAddInteger("nbody", "soft_mode", 0); // 0 = plummer, 1 = spline

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

  // determine array sizes for registers including ghosts
  auto &indcs       = pmy_pack->pmesh->mb_indcs;
  const int ncells1 = indcs.nx1 + 2 * (indcs.ng);
  const int ncells2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*(indcs.ng)) : 1;
  const int ncells3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*(indcs.ng)) : 1;
  const int nmb_max = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));

  // determine gravity module 
  grav_mode = pin->GetOrAddInteger("nbody", "grav_mode", 0);
  switch (grav_mode) {
    case LIVE_CALC: // grav_mode = 0, live calc cell-centered acceleration
      break;
    case PRE_POTENTIAL: // grav_mode = 1, pre-calc potential at centers
      Kokkos::realloc(phi, nmb_max, num_nbody, ncells3, ncells2, ncells1);
      break;
    case PRE_ACCELERATION: // grav_mode = 2, pre-calc acceleration at centers and faces
      // realloc each 3D struct (gv, gf)
      Kokkos::realloc(gv.x1f, nmb_max, num_nbody, ncells3, ncells2, ncells1);
      Kokkos::realloc(gv.x2f, nmb_max, num_nbody, ncells3, ncells2, ncells1);
      Kokkos::realloc(gv.x3f, nmb_max, num_nbody, ncells3, ncells2, ncells1);
      Kokkos::realloc(gf.x1f, nmb_max, num_nbody, ncells3, ncells2, ncells1);
      Kokkos::realloc(gf.x2f, nmb_max, num_nbody, ncells3, ncells2, ncells1);
      Kokkos::realloc(gf.x3f, nmb_max, num_nbody, ncells3, ncells2, ncells1);
      break;
  } // end grav_mode switch
  
  // determine viscosity usage
  alpha     = pin->GetOrAddReal("nbody", "alpha", 0.0);
  visc_mode = pin->GetOrAddInteger("nbody", "visc_mode", 0);
  nu_max    = pin->GetOrAddReal("nbody", "nu_max", 1e-3);
  if (alpha != 0.0) {
    // allocate array for per-cell viscosity coefficient, including ghosts on each side
    Kokkos::realloc(nu_iso, nmb_max, ncells3, ncells2, ncells1); 
    // allocate register for tracking maximal viscosity
    Kokkos::realloc(max_nu_iso, nmb_max);
  } 

  // check validity of user input
  if (global_variable::my_rank == 0) {
    if ((sink_mode == 1) && (pin->GetOrAddReal("hydro","nu_iso",0.0) == 0)) {
      std::cout << "### FATAL ERROR in NBody::NBody" << std::endl
                << "sink_mode = 1 incompatible with nu_iso = 0" << std::endl;
      std::exit(EXIT_FAILURE);
    } else if ((sink_mode == 2) && (alpha == 0)) {
      std::cout << "### FATAL ERROR in NBody::NBody" << std::endl
                << "sink_mode = 2 incompatible with alpha = 0" << std::endl;
      std::exit(EXIT_FAILURE);
    } // end sink_mode check
  } // end if rank 0
  
  // resize principle registers for per-body data                Enumerator for human-readable labelling
  Kokkos::realloc(nbody_data,       num_nbody, NVAR_DATA);    // NBodyDataIndices
  Kokkos::realloc(nbody_comm,       num_nbody, NVAR_DATA);    // NBodyDataIndices
  Kokkos::realloc(nbody_data1,      num_nbody, NVAR_REG);     // NBodyRegisterIndices   
  Kokkos::realloc(nbody_flux,       num_nbody, NVAR_REG);     // NBodyRegisterIndices
  Kokkos::realloc(delta_this_pack,  num_nbody, NVAR_BACK);    // NBodyBackIndices
  Kokkos::realloc(delta_this_mesh,  num_nbody, NVAR_BACK);    // NBodyBackIndices
  Kokkos::realloc(delta_all_meshes, num_nbody, NVAR_BACK);    // NBodyBackIndices
  
  // identify number of AMR levels
  num_levels = pin->GetOrAddInteger("mesh_refinement", "num_levels", 0);
  const int max_level = (num_levels >= 1) ? num_levels - 1 : 0;

  // load initial nbody state from user input
  for (int n = 0; n < num_nbody; n++) {
    // select nth nbody state 
    std::string str_n = std::to_string(n);
    // mass, position and velocity data MUST be passed
    nbody_data.h_view(n, M_DATA)      = pin->GetReal("nbody", "m" + str_n);
    nbody_data.h_view(n, X_DATA)      = pin->GetReal("nbody", "x" + str_n);
    nbody_data.h_view(n, Y_DATA)      = pin->GetReal("nbody", "y" + str_n);
    nbody_data.h_view(n, Z_DATA)      = pin->GetReal("nbody", "z" + str_n);
    nbody_data.h_view(n, VX_DATA)     = pin->GetReal("nbody", "vx" + str_n);
    nbody_data.h_view(n, VY_DATA)     = pin->GetReal("nbody", "vy" + str_n);
    nbody_data.h_view(n, VZ_DATA)     = pin->GetReal("nbody", "vz" + str_n);
    nbody_data.h_view(n, R_SOFT_DATA) = pin->GetReal("nbody", "r_soft" + str_n);
    // amr properties default to zero
    nbody_data.h_view(n, R_AMR_DATA)  = pin->GetOrAddReal("nbody", "r_amr" + str_n, 0.0);
    nbody_data.h_view(n, T_AMR_DATA)  = pin->GetOrAddReal("nbody", "t_amr" + str_n, 0.0);
    nbody_data.h_view(n, N_AMR_DATA)  = pin->GetOrAddInteger("nbody", "n_amr" + str_n, max_level); 
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

  // if verbose and principle NBody instance, write status to command line
  // TODO: package this as report func
  // TODO: this is inacitve, find better catch for t=0 init
  if ((global_variable::my_rank == 0) && (pmy_pack != &pmy_pack->pmesh->pmb_pack[0]) && (pmy_pack->pmesh->time == 0)){
    std::cout << "Reporting NBody init..." << std::endl;
    
    std::cout << "n-body settings:"     << std::endl;
    std::cout << "num_nbody = "         << num_nbody << std::endl;
    std::cout << "inc_backreaction = "  << inc_backreaction << std::endl;

    std::cout << "hydro settings:"      << std::endl;
    std::cout << "src_gravity = "       << src_gravity << std::endl;
    std::cout << "grav_mode = "         << grav_mode << std::endl;
    std::cout << "src_accretion = "     << src_accretion << std::endl;
    std::cout << "cooling_mode = "      << cooling_mode << std::endl;
    std::cout << "sink_mode = "         << sink_mode << std::endl;
    std::cout << "alpha = "             << alpha << std::endl;
  }

} // end ctor

//----------------------------------------------------------------------------------------
//! \fn  void NBody::~NBody
//! \brief Destructor, frees any memory allocated during construction. Currently empty
NBody::~NBody() {
}

//----------------------------------------------------------------------------------------
//! \fn  Real NBody::NBodySrcTerms
//! \brief Compute the maximum timestep for stable NBody evolution given the current 
//! nbody_data state. Collected with other timesteps in Mesh::NewTimeStep 
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

//----------------------------------------------------------------------------------------
//! \fn  void NBody::NBodySrcTerms
//! \brief Wrapper to invoke source terms associated with the NBody class, including
//! runtime booleans to activate specific physical processes
void NBody::NBodySrcTerms(const Real beta_dt) {
  
  // apply gravity source terms if flagged
  if (src_gravity) {
    switch (grav_mode) {
      case LIVE_CALC:
        NBodyPointSrcTerm(beta_dt);             // compute and apply gravity in one par_for execution
        break;
      case PRE_POTENTIAL:
        CalcGravitationalPotential(beta_dt);    // pre-compute per-cell gravitational potential
        ApplyGravitationalPotential(beta_dt);   // apply pre-computed gravity
        break;
      case PRE_ACCELERATION:
        CalcGravitationalAcceleration(beta_dt);    // pre-compute per-cell gravitational potential
        ApplyGravitationalAcceleration(beta_dt);   // apply pre-computed gravity
        break;
    } // end grav_mode cases
  } // end src_gravity

  // apply accretion, if flagged
  if (src_accretion) {
    if (grav_mode != LIVE_CALC) { // if LIVE_CALC, accretion already included in NBodyPointSrcTerm
      ApplyAccretion(beta_dt);
    } // end grav_mode check
  } // end src_accretion
  

  // thermodynamic src terms
  if (pmy_pack->phydro->peos->eos_data.is_ideal) {
    switch (cooling_mode) {
      case NO_COOLING:      // no cooling term flagged, skip
        break;
      case FORCED_ISO:      // instantly fix temperature to fixed Mach number
        NBodyForcedIsoSrcTerm(beta_dt);
        break;
      case BLACK_BODY:      // apply optically thick black body cooling
        NBodyBlackBodySrcTerm(beta_dt);
        break;
      case BETA_COOL_DYN:  // apply beta cooling to fixed Mach number on dynamical timescale
        NBodyBetaCoolSrcTerm(beta_dt);
        break;
      case BETA_COOL_THERM: // apply beta cooling to fixed Mach number on thermal timescale
        NBodyBetaCoolSrcTerm(beta_dt);
        break;
      default:
        break;
    } // end cooling switch
  } // end is_ideal

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::CalcGravitationalPotential
//! \brief Populate an array of gravitational potentials on a per-cell basis to be used 
//! for calculating momentum/energy updates in NBody::ApplyGravitationalPotential
void NBody::CalcGravitationalPotential(const Real beta_dt) {

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

  // general NBody properties 
  auto &nbody_data_    = nbody_data;
  const int num_nbody_ = num_nbody;
  const Real G_const_  = G_const;
  auto &phi_           = phi;

  par_for("nbody_calc_potential", 
          DevExeSpace(), 
          0, nmb1, 
          ks, ke, 
          js, je, 
          is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // identify cell face position
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // loop over bodies in NBody class
      for (int n = 0; n < num_nbody_; n++) {
        // compute body-cell seperation
        const Real dx = x - nbody_data_.d_view(n, X_DATA);
        const Real dy = y - nbody_data_.d_view(n, Y_DATA);
        const Real dz = z - nbody_data_.d_view(n, Z_DATA);
        const Real dr_sqr_soft = SQR(dx) + SQR(dy) + SQR(dz) + SQR(nbody_data_.d_view(n, R_SOFT_DATA));

        // compute and set softened gravitational potential
        const Real phi_local = G_const_ * nbody_data_.d_view(n, M_DATA) * Kokkos::pow(dr_sqr_soft, -0.5);
        phi_(mb_id, n, k, j, i) = phi_local;
      } // end body loop
    }); // end par_for
    
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::CalcGravitationalPotential
//! \brief Populate an array of gravitational potentials on a per-cell basis to be used 
//! for calculating momentum/energy updates in NBody::ApplyGravitationalPotential
void NBody::CalcGravitationalAcceleration(const Real beta_dt) {

  // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  // par_for includes ghost cells in active directions (dimension sensitive)
  bool &multi_d = pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_pack->pmesh->three_d;
  int il = indcs.is - indcs.ng, ir = indcs.ie + indcs.ng;
  int jl = indcs.js, jr = indcs.je;
  int kl = indcs.ks, kr = indcs.ke;
  if (multi_d) {
    jl -= indcs.ng; jr += indcs.ng;
  }
  if (three_d) {
    kl -= indcs.ng; kr += indcs.ng;
  }
  int nmb1 = pmy_pack->nmb_thispack - 1;
  
  // general NBody properties 
  auto &nbody_data_    = nbody_data;
  const int num_nbody_ = num_nbody;
  const Real G_const_  = G_const;
  const Real soft_mode_ = soft_mode;

  // unpack gravity registers
  // registers are DvceFaceFld5D objects, but for gv, x1f labels the cell-centered acceleration in x1
  // for gv, x1f labels the lower-x face-centered acceleration in x1
  auto &gv_x           = gv.x1f;
  auto &gv_y           = gv.x2f;
  auto &gv_z           = gv.x3f;
  auto &gf_x           = gf.x1f;
  auto &gf_y           = gf.x2f;
  auto &gf_z           = gf.x3f;

  // iterate over single ghost on either side of real domain to ensure evaluation on all faces (i,i+1)
  par_for("nbody_calc_acceleration", 
          DevExeSpace(), 
          0, nmb1, 
          kl, kr, 
          jl, jr, 
          il, ir,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // identify cell center positions
      const Real xv = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real yv = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real zv = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // identify lower cell face positions 
      const Real xf = LeftEdgeX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real yf = LeftEdgeX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real zf = LeftEdgeX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // loop over bodies in NBody class
      for (int n = 0; n < num_nbody_; n++) {
        // package body properties
        const Real Gm_n = G_const_ * nbody_data_.d_view(n, M_DATA);
        const Real r_s  = nbody_data_.d_view(n, R_SOFT_DATA);
        
        // compute gravity at cell center
        const Real dxv = xv - nbody_data_.d_view(n, X_DATA);
        const Real dyv = yv - nbody_data_.d_view(n, Y_DATA);
        const Real dzv = zv - nbody_data_.d_view(n, Z_DATA);
        const Real drv = Kokkos::sqrt(SQR(dxv) + SQR(dyv) + SQR(dzv)); // cc_x, cc_y, cc_z
        Real gv_kji;
        if (soft_mode_ == 0) {
          gv_kji = CalcPlummerAcceleration(Gm_n, drv, r_s);
        } else {
          gv_kji = CalcSplineAcceleration(Gm_n, drv, r_s);
        }

        // project components of cell-centered gravity in cardinal directions
        const Real gv_over_drv  = gv_kji / drv;
        gv_x(mb_id, n, k, j, i) = gv_over_drv * dxv;
        if (multi_d) gv_y(mb_id, n, k, j, i) = gv_over_drv * dyv; 
        if (three_d) gv_z(mb_id, n, k, j, i) = gv_over_drv * dzv;

        // compute gravity on lower x face
        {
          const Real dxf  = xf - nbody_data_.d_view(n, X_DATA);
          const Real drfx = Kokkos::sqrt(SQR(dxf) + SQR(dyv) + SQR(dzv)); // fc_x, cc_y, cc_z
          Real gfx_kji; 
          if (soft_mode_ == 0) {
            gfx_kji = CalcPlummerAcceleration(Gm_n, drfx, r_s);
          } else {
            gfx_kji = CalcSplineAcceleration(Gm_n, drfx, r_s);
          }
          gf_x(mb_id, n, k, j, i) = gfx_kji * dxf / drfx; // only stash the component perp to face
        }

        // compute gravity on lower y face
        if (multi_d) {
          const Real dyf  = yf - nbody_data_.d_view(n, Y_DATA);
          const Real drfy = Kokkos::sqrt(SQR(dxv) + SQR(dyf) + SQR(dzv)); // cc_x, fc_y, cc_z
          Real gfy_kji; 
          if (soft_mode_ == 0) {
            gfy_kji = CalcPlummerAcceleration(Gm_n, drfy, r_s);
          } else {
            gfy_kji = CalcSplineAcceleration(Gm_n, drfy, r_s);
          }
          gf_y(mb_id, n, k, j, i) = gfy_kji * dyf / drfy; // only stash the component perp to face
        } // end multi_d

        // compute gravity on lower z face
        if (three_d) {
          const Real dzf  = zf - nbody_data_.d_view(n, Z_DATA);
          const Real drfz = Kokkos::sqrt(SQR(dxv) + SQR(dyv) + SQR(dzf)); // cc_x, cc_y, fc_z
          Real gfz_kji; 
          if (soft_mode_ == 0) {
            gfz_kji = CalcPlummerAcceleration(Gm_n, drfz, r_s);
          } else {
            gfz_kji = CalcSplineAcceleration(Gm_n, drfz, r_s);
          }
          gf_z(mb_id, n, k, j, i) = gfz_kji * dzf / drfz; // only stash the component perp to face
        } // end three_d
      } // end body loop
    }); // end par_for
    
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::ApplyGravitationalPotential
//! \brief Following pre-computation of the by-cell gravitational potential, compute 
//! gravitational accelerations and apply momentum/energy updates to all cells.
void NBody::ApplyGravitationalPotential(const Real beta_dt) {

  // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &prim = pmy_pack->phydro->w0;
  auto &cons = pmy_pack->phydro->u0;
  const bool three_d = pmy_pack->pmesh->three_d;

  // general NBody properties 
  const int num_nbody_ = num_nbody;
  auto &phi_           = phi;

  // Get Godunov density fluxes from Riemann solver
  DvceArray5D<Real> flx1, flx2, flx3;
  flx1 = pmy_pack->phydro->uflx.x1f;
  flx2 = pmy_pack->phydro->uflx.x2f;
  flx3 = pmy_pack->phydro->uflx.x3f;

  // thermodynamic properties
  const bool evolve_energy  = (pmy_pack->phydro->peos->eos_data.is_ideal) && (cooling_mode != FORCED_ISO);

  // back reation properties
  bool sum_backreaction_ = sum_backreaction;
  auto &delta_this_pack_ = delta_this_pack;
  
  par_for("nbody_apply_potential", 
          DevExeSpace(), 
          0, nmb1, 
          ks, ke, 
          js, je, 
          is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // compute cell seperations
      const Real m1_odx1 = -1.0 / size.d_view(mb_id).dx1;
      const Real m1_odx2 = -1.0 / size.d_view(mb_id).dx2;
      const Real m1_odx3 = -1.0 / size.d_view(mb_id).dx3;

      // loop over bodies in NBody class
      for (int n = 0; n < num_nbody_; n++) {
        // compute acceleration on cell faces using pre-calc values
        const Real g_xl = m1_odx1 * (phi_(mb_id, n, k, j, i) - phi_(mb_id, n, k, j, i - 1));
        const Real g_xr = m1_odx1 * (phi_(mb_id, n, k, j, i + 1) - phi_(mb_id, n, k, j, i));
        const Real g_yl = m1_odx2 * (phi_(mb_id, n, k, j, i) - phi_(mb_id, n, k, j - 1, i));
        const Real g_yr = m1_odx2 * (phi_(mb_id, n, k, j + 1, i) - phi_(mb_id, n, k, j, i));
        Real g_zl = 0.0, g_zr = 0.0;
        if (three_d) {
          const Real g_zl = m1_odx3 * (phi_(mb_id, n, k, j, i) - phi_(mb_id, n, k - 1, j, i));
          const Real g_zr = m1_odx3 * (phi_(mb_id, n, k + 1, j, i) - phi_(mb_id, n, k, j, i));
        }

        // compute cell centered accelerations
        const Real g_xv = 0.5 * (g_xl + g_xr);
        const Real g_yv = 0.5 * (g_yl + g_yr);
        const Real g_zv = 0.5 * (g_zl + g_zr); // zero if !three_d

        // update momenta
        const Real rho_beta_dt = prim(mb_id, IDN, k, j, i) * beta_dt; 
        const Real dpx  = g_xv * rho_beta_dt;
        const Real dpy  = g_yv * rho_beta_dt;
        const Real dpz  = g_zv * rho_beta_dt;
        cons(mb_id, IM1, k, j, i) += dpx;
        cons(mb_id, IM2, k, j, i) += dpy;
        if (three_d) cons(mb_id, IM3, k, j, i) += dpz;

        // update energy
        if (evolve_energy) {
          Real rho_v_dot_g          = 0.5 * (flx1(mb_id, IDN, k, j, i) * g_xl + flx1(mb_id, IDN, k, j, i + 1) * g_xr);
          rho_v_dot_g              += 0.5 * (flx2(mb_id, IDN, k, j, i) * g_yl + flx2(mb_id, IDN, k, j + 1, i) * g_yr);
          if (three_d) rho_v_dot_g += 0.5 * (flx3(mb_id, IDN, k, j, i) * g_zl + flx3(mb_id, IDN, k + 1, j, i) * g_zr);
          cons(mb_id,IEN,k,j,i)    += rho_v_dot_g * beta_dt;
        } // end energy update

        // compute backreaction 
        if (sum_backreaction_) {
          // convert from densities to absolute changes
          const Real cell_volume = size.d_view(mb_id).dx1 * size.d_view(mb_id).dx2 * size.d_view(mb_id).dx3;
          // compute momentum change for gravitating body
          const Real dPx = -dpx * cell_volume;
          const Real dPy = -dpy * cell_volume;
          const Real dPz = -dpz * cell_volume;
          // stash backreaction registers, with care for race conditions
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPX_GRAV_BACK), dPx);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPY_GRAV_BACK), dPy);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPZ_GRAV_BACK), dPz);
        } // end backreaction
      } // end body loop
    }); // end par_for
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::ApplyAccretion
//! \brief Apply source term relevant for accretion
void NBody::ApplyAccretion(const Real beta_dt) {

  // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &prim = pmy_pack->phydro->w0;
  auto &cons = pmy_pack->phydro->u0;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  // general NBody properties 
  auto &nbody_data_     = nbody_data;
  const int num_nbody_  = num_nbody;
  const Real G_const_   = G_const;

  // thermodynamic properties
  const bool evolve_energy  = (pmy_pack->phydro->peos->eos_data.is_ideal) && (cooling_mode != FORCED_ISO);
  const Real inv_Mach_sqr_ = inv_Mach_sqr;
  const Real gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  const Real inv_gm1 = 1.0 / gm1;

  // accretion properties
  const Real rho_sink_floor_ = rho_sink_floor;
  const int sink_mode_       = sink_mode;
  Real nu_iso_homo_          = 0.0;
  if (sink_mode_ == 1) nu_iso_homo_ = pmy_pack->phydro->pvisc->nu_iso;
  auto &nu_iso_inhomo_       = nu_iso;
  Real alpha_ = (alpha != 0.0) ? alpha : 0.1; // use fiducial value if not set 
  const Real max_sink_fac = 0.5; // maximal fractional loss per fine timestep

  // back reation properties
  bool sum_backreaction_ = sum_backreaction;
  auto &delta_this_pack_ = delta_this_pack;

  par_for("nbody_apply_accretion", 
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
        const Real dx     = x - nbody_data_.d_view(n, X_DATA);
        const Real dy     = y - nbody_data_.d_view(n, Y_DATA);
        const Real dz     = z - nbody_data_.d_view(n, Z_DATA);
        const Real dr_sqr = SQR(dx) + SQR(dy) + SQR(dz);

        // make proximity check and floor check
        const Real r_soft     = nbody_data_.d_view(n, R_SOFT_DATA);
        const Real drho_floor = rho_sink_floor_ - prim(mb_id, IDN, k, j, i);
        if ((dr_sqr < 2 * SQR(r_soft)) && (drho_floor < 0)) {
          Real drho_acc = 0, dpx_acc = 0, dpy_acc = 0, dpz_acc = 0;
          const Real dr = Kokkos::sqrt(dr_sqr);
          // compute mass removal rate
          Real sink_rate = 0.0; // defined such that drho/dt = -rho * sink_rate
          if (sink_mode_ == FLAT_RATE) {          // set to fixed, high value
            sink_rate = 100;
          } else if (sink_mode_ == HOMO_VISC) {   // set by viscous time, using homogeneous viscosity
            sink_rate = 2.25 * nu_iso_homo_ / dr_sqr;
          } else if (sink_mode_ == INHOMO_VISC) { // set by viscous time, using inhomogeneous viscosity
            sink_rate = 2.25 * nu_iso_inhomo_(mb_id, k, j, i) / dr_sqr;
          } else if (sink_mode_ == IDEAL_VISC) {  // set by viscous time, using ideal fixed Mach disc
            const Real nu_ideal = alpha_ * inv_Mach_sqr_ * Kokkos::sqrt(G_const_ * nbody_data_.d_view(n, M_DATA) * dr);
            sink_rate = 2.25 * nu_ideal / dr_sqr;
          } else if (sink_mode_ == ADAPT_VISC) {  // set by viscous time, using local potential by adaptive sound speed
            const Real cs_sqr_local = prim(mb_id, IEN, k, j, i) * gm1 / prim(mb_id, IDN, k, j, i);
            const Real omega_sqr_local = G_const_ * nbody_data_.d_view(n, M_DATA) * Kokkos::pow(dr, -3.0);
            const Real nu_local = alpha_ * cs_sqr_local * Kokkos::pow(omega_sqr_local, -0.5);
            sink_rate = 2.25 * nu_local / dr_sqr;
          } // end sink_mode cases

          // apply window and ceiling to sink rate
          sink_rate *= Kokkos::exp(-Kokkos::pow(dr / r_soft, 4));      // apply sharpened Gaussian profile to sink region
          sink_rate  = Kokkos::min(sink_rate, max_sink_fac / beta_dt); // limit to maximum % removal per finetimestep
          const Real rhodot = - prim(mb_id, IDN, k, j, i) * sink_rate;
          drho_acc = rhodot * beta_dt;
          drho_acc = Kokkos::max(drho_acc, drho_floor);                // ensure accretion only drives to sink value

          // determine momentum change, given torque free sink TODO: add option for non-torque free sink
          const Real inv_dr = 1.0 / dr;
          const Real rhatx = dx * inv_dr;
          const Real rhaty = dy * inv_dr;
          const Real rhatz = dz * inv_dr;
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

          // apply accretion updates to cell's conserved quantities
          cons(mb_id, IDN, k, j, i) += drho_acc;
          cons(mb_id, IM1, k, j, i) += dpx_acc;
          if (multi_d) cons(mb_id, IM2, k, j, i) += dpy_acc;
          if (three_d) cons(mb_id, IM3, k, j, i) += dpz_acc;

          // adiabatic update from accretion
          if (evolve_energy) {
            // compute current specific internal energy and kinetic energy updates
            const Real epsilon        = prim(mb_id, IEN, k, j, i) / prim(mb_id, IDN, k, j, i);
            const Real vstar_sqr      = SQR(vxstar) + SQR(vystar) + SQR(vzstar);
            const Real vdagger_sqr    = SQR(prim(mb_id, IVX, k, j, i) - vxstar) 
                                      + SQR(prim(mb_id, IVY, k, j, i) - vystar) 
                                      + SQR(prim(mb_id, IVZ, k, j, i) - vzstar);
            const Real dE_acc         = drho_acc * (epsilon + 0.5 * vstar_sqr - 0.5 * vdagger_sqr);
            cons(mb_id, IEN, k, j, i) += dE_acc;
          } // end energy update

          // if flagged, report backreaction on body 
          if (sum_backreaction_) {
            // convert from densities 
            const Real cell_volume = size.d_view(mb_id).dx1 * size.d_view(mb_id).dx2 * size.d_view(mb_id).dx3;
            const Real dm_tot   = -drho_acc * cell_volume; 
            const Real dPx_acc  = -dpx_acc  * cell_volume;
            const Real dPy_acc  = -dpy_acc  * cell_volume;
            const Real dPz_acc  = -dpz_acc  * cell_volume;
            // make thread-safe add to backreaction registers
            Kokkos::atomic_add(&delta_this_pack_.d_view(n, DM_BACK), dm_tot);
            Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPX_ACC_BACK), dPx_acc);
            Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPY_ACC_BACK), dPy_acc);
            Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPZ_ACC_BACK), dPz_acc);
          } // end backreaction
        } // end if flagged for accretion
      } // end body loop
    }); // end par_for

  return;
} // end NBody::ApplyAccretion

//----------------------------------------------------------------------------------------
//! \fn  void NBody::ApplyGravitationalAcceleration
//! \brief Following pre-computation of the by-cell gravitational acceleration, compute 
//! momentum/energy updates for all cells.
void NBody::ApplyGravitationalAcceleration(const Real beta_dt) {

    // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &prim = pmy_pack->phydro->w0;
  auto &cons = pmy_pack->phydro->u0;
  const bool multi_d = pmy_pack->pmesh->multi_d;
  const bool three_d = pmy_pack->pmesh->three_d;

  // general NBody properties 
  const int num_nbody_ = num_nbody;

  // unpack gravity registers
  auto &gv_x           = gv.x1f;
  auto &gv_y           = gv.x2f;
  auto &gv_z           = gv.x3f;
  auto &gf_x           = gf.x1f;
  auto &gf_y           = gf.x2f;
  auto &gf_z           = gf.x3f;

  // unpack Godunov density fluxes from Riemann solver
  DvceArray5D<Real> flx1, flx2, flx3;
  if (pmy_pack->pmhd != nullptr) {
    flx1 = pmy_pack->pmhd->uflx.x1f;
    flx2 = pmy_pack->pmhd->uflx.x2f;
    flx3 = pmy_pack->pmhd->uflx.x3f;
  } else {
    flx1 = pmy_pack->phydro->uflx.x1f;
    flx2 = pmy_pack->phydro->uflx.x2f;
    flx3 = pmy_pack->phydro->uflx.x3f;
  }

  // thermodynamic properties
  const bool evolve_energy  = (pmy_pack->phydro->peos->eos_data.is_ideal) && (cooling_mode != FORCED_ISO);

  // back reation properties
  bool sum_backreaction_ = sum_backreaction;
  auto &delta_this_pack_ = delta_this_pack;
  
  par_for("nbody_apply_acceleration", 
          DevExeSpace(), 
          0, nmb1, 
          ks, ke, 
          js, je, 
          is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // loop over bodies in NBody class
      for (int n = 0; n < num_nbody_; n++) {
        // update momenta using cell-centered acceleration (gv) and primitive density
        const Real rho_beta_dt = prim(mb_id, IDN, k, j, i) * beta_dt; 
        const Real dpx         = gv_x(mb_id, n, k, j, i) * rho_beta_dt;
        const Real dpy         = gv_y(mb_id, n, k, j, i) * rho_beta_dt;
        const Real dpz         = gv_z(mb_id, n, k, j, i) * rho_beta_dt;
        cons(mb_id, IM1, k, j, i) += dpx;
        if (multi_d) cons(mb_id, IM2, k, j, i) += dpy;
        if (three_d) cons(mb_id, IM3, k, j, i) += dpz;

        // update energy using face-centered acceleration and Godunov mass flux on face
        // i = lower x face, i + 1 = upper x face (and same for y, z with j, k)
        if (evolve_energy) {
          // // METHOD 1: CC ONLY - this is stable and matches the NBodyPointSrc behaviour
          Real v_dot_g = prim(mb_id, IVX, k, j, i) * gv_x(mb_id, n, k, j, i);
          if (multi_d) v_dot_g += prim(mb_id, IVY, k, j, i) * gv_y(mb_id, n, k, j, i);
          if (three_d) v_dot_g += prim(mb_id, IVZ, k, j, i) * gv_z(mb_id, n, k, j, i);
          const Real rho_v_dot_g = prim(mb_id, IDN, k, j, i) * v_dot_g;
          cons(mb_id, IEN, k, j, i) += rho_v_dot_g * beta_dt;

          // const Real dEx_fc = 0.5 * (flx1(mb_id, IDN, k, j, i)     * gf_x(mb_id, n, k, j, i) 
          //                          + flx1(mb_id, IDN, k, j, i + 1) * gf_x(mb_id, n, k, j, i + 1));
          // const Real dEx_cc = prim(mb_id, IVX, k, j, i) * prim(mb_id, IDN, k, j, i) * gv_x(mb_id, n, k, j, i);

          // const Real dEy_fc = 0.5 * (flx2(mb_id, IDN, k, j, i)     * gf_y(mb_id, n, k, j, i) 
          //                          + flx2(mb_id, IDN, k, j + 1, i) * gf_y(mb_id, n, k, j + 1, i));
          // const Real dEy_cc = prim(mb_id, IVY, k, j, i) * prim(mb_id, IDN, k, j, i) * gv_y(mb_id, n, k, j, i);
          //Kokkos::printf("ratio_x = %f, ratio_y = %f\n", dEx_fc / dEx_cc, dEy_fc / dEy_cc);

          // METHOD 2: use fluxes, but only cc gravity - produces the same energy errors as Method 3
          // Real rho_v_dot_g           = 0.5 * (flx1(mb_id, IDN, k, j, i)     * gv_x(mb_id, n, k, j, i) 
          //                                   + flx1(mb_id, IDN, k, j, i + 1) * gv_x(mb_id, n, k, j, i));
          // if (multi_d) rho_v_dot_g  += 0.5 * (flx2(mb_id, IDN, k, j, i)     * gv_y(mb_id, n, k, j, i) 
          //                                   + flx2(mb_id, IDN, k, j + 1, i) * gv_y(mb_id, n, k, j, i)); 
          // if (three_d) rho_v_dot_g  += 0.5 * (flx3(mb_id, IDN, k, j, i)     * gv_z(mb_id, n, k, j, i) 
          //                                   + flx3(mb_id, IDN, k + 1, j, i) * gv_z(mb_id, n, k, j, i)); 
          // cons(mb_id, IEN, k, j, i) += rho_v_dot_g * beta_dt;

          // METHOD 3: full conservative model
          // Real rho_v_dot_g           = 0.5 * (flx1(mb_id, IDN, k, j, i)     * gf_x(mb_id, n, k, j, i) 
          //                                   + flx1(mb_id, IDN, k, j, i + 1) * gf_x(mb_id, n, k, j, i + 1));
          // if (multi_d) rho_v_dot_g  += 0.5 * (flx2(mb_id, IDN, k, j, i)     * gf_y(mb_id, n, k, j, i) 
          //                                   + flx2(mb_id, IDN, k, j + 1, i) * gf_y(mb_id, n, k, j + 1, i)); 
          // if (three_d) rho_v_dot_g  += 0.5 * (flx3(mb_id, IDN, k, j, i)     * gf_z(mb_id, n, k, j, i) 
          //                                   + flx3(mb_id, IDN, k + 1, j, i) * gf_z(mb_id, n, k + 1, j, i)); 
          // cons(mb_id, IEN, k, j, i) += rho_v_dot_g * beta_dt;
        } // end energy update

        // compute backreaction 
        if (sum_backreaction_) {
          // convert from momentum densities to absolute momentum using cell volume
          const Real cell_volume = size.d_view(mb_id).dx1 * size.d_view(mb_id).dx2 * size.d_view(mb_id).dx3;
          // compute momentum change for gravitating body
          const Real dPx = -dpx * cell_volume;
          const Real dPy = -dpy * cell_volume;
          const Real dPz = -dpz * cell_volume;
          // stash backreaction registers, with care for race conditions
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPX_GRAV_BACK), dPx);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPY_GRAV_BACK), dPy);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPZ_GRAV_BACK), dPz);
        } // end backreaction
      } // end body loop
    }); // end par_for
  return;
} // end NBody::ApplyGravitationalAcceleration

//----------------------------------------------------------------------------------------
//! \fn  void NBody::NBodyPointSrcTerm
//! \brief Includes any source terms that operate on a per-body basis, namely gravity
//! and accretion. Stash backreaction effects such as mass, momentum changes for bodies.
void NBody::NBodyPointSrcTerm(const Real beta_dt) {

  // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &prim = pmy_pack->phydro->w0;
  auto &cons = pmy_pack->phydro->u0;

  // general NBody properties 
  auto &nbody_data_     = nbody_data;
  const int num_nbody_  = num_nbody;
  const Real G_const_   = G_const;
  const Real soft_mode_ = soft_mode;

  // thermodynamic properties
  const bool is_ideal       = pmy_pack->phydro->peos->eos_data.is_ideal;
  const bool evolve_energy  = (cooling_mode != FORCED_ISO);

  // back reation properties
  bool sum_backreaction_ = sum_backreaction;
  auto &delta_this_pack_ = delta_this_pack;
  
  // accretion properties
  const bool src_accretion_  = src_accretion;
  const Real rho_sink_floor_ = rho_sink_floor;
  const int sink_mode_       = sink_mode;
  Real nu_iso_homo_          = 0.0;
  if (sink_mode_ == 1) nu_iso_homo_ = pmy_pack->phydro->pvisc->nu_iso;
  auto &nu_iso_inhomo_       = nu_iso;
  Real alpha_ = (alpha != 0.0) ? alpha : 0.1; // use fiducial value if not set 
  Real inv_Mach_sqr_ = inv_Mach_sqr;
  const Real gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;
  const Real inv_gm1 = 1.0 / gm1;
  const Real max_sink_fac = 0.5; // maximal fractional loss per fine timestep

  // capture flux arrays for energy updates
  DvceArray5D<Real> flx1, flx2, flx3;
  if (pmy_pack->pmhd != nullptr) {
    flx1 = pmy_pack->pmhd->uflx.x1f;
    flx2 = pmy_pack->pmhd->uflx.x2f;
    flx3 = pmy_pack->pmhd->uflx.x3f;
  } else {
    flx1 = pmy_pack->phydro->uflx.x1f;
    flx2 = pmy_pack->phydro->uflx.x2f;
    flx3 = pmy_pack->phydro->uflx.x3f;
  }

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
        const Real dx     = x - nbody_data_.d_view(n, X_DATA);
        const Real dy     = y - nbody_data_.d_view(n, Y_DATA);
        const Real dz     = z - nbody_data_.d_view(n, Z_DATA);
        const Real dr_sqr = SQR(dx) + SQR(dy) + SQR(dz);
        const Real dr     = Kokkos::sqrt(dr_sqr);
        const Real inv_dr = 1.0 / dr;
        const Real r_soft = nbody_data_.d_view(n, R_SOFT_DATA);
        const Real Gm_n   = G_const_ * nbody_data_.d_view(n, M_DATA);

        // compute softened Newtonian gravitational acceleration
        const Real rho = prim(mb_id, IDN, k, j, i);
        Real g_soft;
        if (soft_mode_ == 0) {
          g_soft = CalcPlummerAcceleration(Gm_n, dr, r_soft);
        } else {
          g_soft = CalcSplineAcceleration(Gm_n, dr, r_soft);
        }
        
        // compute momentum updates from gravity acceleration
        const Real dp_grav_fac = g_soft * beta_dt * rho * inv_dr;
        const Real dpx_grav    = dp_grav_fac * dx;
        const Real dpy_grav    = dp_grav_fac * dy;
        const Real dpz_grav    = dp_grav_fac * dz;

        // apply gravity updates to cell's conserved quantities
        cons(mb_id, IM1, k, j, i) += dpx_grav;
        cons(mb_id, IM2, k, j, i) += dpy_grav;
        cons(mb_id, IM3, k, j, i) += dpz_grav;

        if (is_ideal && evolve_energy) { // only compute energy change if ideal AND not forced iso
          // gravity update applied as work done W = rho v dot g -> dW = dp dot v
          // TEMP: for simplicity, assume acceleration at faces equal to acceleration at center
          const Real dE_grav = dpx_grav * prim(mb_id, IVX, k, j, i)
                             + dpy_grav * prim(mb_id, IVY, k, j, i)
                             + dpz_grav * prim(mb_id, IVZ, k, j, i);
          cons(mb_id, IEN, k, j, i) += dE_grav;
        } // end energy evo

        // apply acceleration updates AFTER gravity update applied
        
        // apply accretion, if in sink radius and flagged
        Real drho_acc = 0, dpx_acc = 0, dpy_acc = 0, dpz_acc = 0;
        if (src_accretion_) {
          // only accrete down to floor value to avoid blowup
          const Real drho_floor = rho_sink_floor_ - rho; 
          const Real r_ratio = dr / r_soft;
          if ((drho_floor < 0) && (r_ratio < 2)) { // only accrete around sink and above floor
            // compute mass loss rate
            Real sink_rate; // defined such that rhodot = -rho * sink_rate
            if (sink_mode_ == 0) {        // use flat, hungry sink rate
              sink_rate = 100;
            } else if (sink_mode_ == 1) { // rate is inverse of viscous time for flat nu (tau_nu = 9/4 r^2/nu)
              sink_rate = 2.25 * nu_iso_homo_ / dr_sqr;
            } else if (sink_mode_ == 2) {  // rate is inverse of viscous time for inhomo nu
              sink_rate = 2.25 * nu_iso_inhomo_(mb_id, k, j, i) / dr_sqr;
            } else if (sink_mode_ == 3) { // rate is inverse of viscous time for idealised nu
              // ideal viscosity nu = alpha * cs^2 * Omega^-1
              //                    = alpha * M^-2 * sqrt{Gmr}          
              const Real nu_ideal = alpha_ * inv_Mach_sqr_ * Kokkos::sqrt(G_const_ * nbody_data_.d_view(n, M_DATA) * dr);
              sink_rate = 2.25 * nu_ideal / dr_sqr;
            } else {                      // rate is inverse of viscous time with adaptive local temp
              // true viscosity nu = alpha * cs^2 * Omega^-1
              //                   = alpha * U * (g - 1) / rho * Omega^-1
              const Real cs_sqr_local = prim(mb_id, IEN, k, j, i) * gm1 / prim(mb_id, IDN, k, j, i);
              const Real omega_sqr_local = G_const_ * nbody_data_.d_view(n, M_DATA) * Kokkos::pow(dr, -3.0);
              const Real nu_local = alpha_ * cs_sqr_local * Kokkos::pow(omega_sqr_local, -0.5);
              sink_rate = 2.25 * nu_local / dr_sqr;
            }
            sink_rate *= Kokkos::exp(-Kokkos::pow(r_ratio, 4));          // apply sharpened Gaussian profile to sink region
            sink_rate  = Kokkos::min(sink_rate, max_sink_fac / beta_dt); // limit to maximum % removal per finetimestep
            const Real rhodot = - rho * sink_rate;
            drho_acc = rhodot * beta_dt;
            // ensure accretion only drives to sink value
            drho_acc = Kokkos::max(drho_acc, drho_floor);
            // apply torque free sink
            const Real rhatx = dx * inv_dr;
            const Real rhaty = dy * inv_dr;
            const Real rhatz = dz * inv_dr;
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

            // apply accretion updates to cell's conserved quantities
            cons(mb_id, IDN, k, j, i) += drho_acc;
            cons(mb_id, IM1, k, j, i) += dpx_acc;
            cons(mb_id, IM2, k, j, i) += dpy_acc;
            cons(mb_id, IM3, k, j, i) += dpz_acc;

            // adiabatic update from accretion
            if (is_ideal && evolve_energy) {
              // compute current specific internal energy and kinetic energy updates
              const Real epsilon        = prim(mb_id, IEN, k, j, i) / prim(mb_id, IDN, k, j, i);
              const Real vstar_sqr      = SQR(vxstar) + SQR(vystar) + SQR(vzstar);
              const Real vdagger_sqr    = SQR(prim(mb_id, IVX, k, j, i) - vxstar) 
                                        + SQR(prim(mb_id, IVY, k, j, i) - vystar) 
                                        + SQR(prim(mb_id, IVZ, k, j, i) - vzstar);
              const Real dE_acc         = drho_acc * (epsilon + 0.5 * vstar_sqr - 0.5 * vdagger_sqr);
              cons(mb_id, IEN, k, j, i) += dE_acc;
            } // end energy evolution
          } // end in sink
        } // end src_accretion 

        // compute backreaction on body 
        if (sum_backreaction_) {
          // convert from densities 
          const Real cell_volume = size.d_view(mb_id).dx1 * size.d_view(mb_id).dx2 * size.d_view(mb_id).dx3;
          
          // gravity backreaction
          const Real dPx_grav = -dpx_grav * cell_volume;
          const Real dPy_grav = -dpy_grav * cell_volume;
          const Real dPz_grav = -dpz_grav * cell_volume;

          // stash backreaction registers, with care for race conditions
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPX_GRAV_BACK), dPx_grav);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPY_GRAV_BACK), dPy_grav);
          Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPZ_GRAV_BACK), dPz_grav);

          // accretion backreaction
          if (src_accretion_) {
            const Real dm_tot   = -drho_acc * cell_volume; 
            const Real dPx_acc  = -dpx_acc  * cell_volume;
            const Real dPy_acc  = -dpy_acc  * cell_volume;
            const Real dPz_acc  = -dpz_acc  * cell_volume;
            Kokkos::atomic_add(&delta_this_pack_.d_view(n, DM_BACK), dm_tot);
            Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPX_ACC_BACK), dPx_acc);
            Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPY_ACC_BACK), dPy_acc);
            Kokkos::atomic_add(&delta_this_pack_.d_view(n, DPZ_ACC_BACK), dPz_acc);
          } // end if acc
        } // end sum backreaction
      } // end body loop
    }); // end par_for
    
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::NBodyForcedIsoSrcTerm
//! \brief Instantaneously force all cells to have a temperature appropriate for a disc
//! with fixed Mach number, as set in the <nbody> athinput block. Acts on the updated 
//! conserved fluid state to ensure that temperature for next timestep is accurate.
void NBody::NBodyForcedIsoSrcTerm(const Real beta_dt) {

  // unpack all data pre par_for

  // MeshBlock properties
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &size  = pmy_pack->pmb->mb_size;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &cons = pmy_pack->phydro->u0;

  // NBody properties
  auto &nbody_data_ = nbody_data;
  //auto &general_data_ = general_data; // TODO track energy lost by cooling
  const int num_nbody_ = num_nbody;
  const Real inv_Mach_sqr_ = inv_Mach_sqr;
  const Real inv_gm1 = 1.0 / (pmy_pack->phydro->peos->eos_data.gamma - 1.0);
  const Real cs_sqr_floor_ = cs_sqr_floor;
  const Real soft_mode_ = soft_mode;
  const Real G_const_  = G_const;

  par_for("nbody_forced_iso_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // identify cell position
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // compute local sound speed via linear sum of gravitational potentials
      Real sum_phi = 0.0;
      for (int n = 0; n < num_nbody_; n++) {
        // compute true and softened seperation
        const Real dx  = x - nbody_data_.d_view(n, X_DATA);
        const Real dy  = y - nbody_data_.d_view(n, Y_DATA);
        const Real dz  = z - nbody_data_.d_view(n, Z_DATA);
        const Real dr  = Kokkos::sqrt(SQR(dx) + SQR(dy) + SQR(dz)) + FLT_MIN;
        const Real m_n = nbody_data_.d_view(n, M_DATA);
        const Real r_s = nbody_data_.d_view(n, R_SOFT_DATA);
        // compute softened gravitational acceleration
        if (soft_mode_ == 0) {
          sum_phi += CalcPlummerPotential(G_const_ * m_n, dr, r_s);
        } else {
          sum_phi += CalcSplinePotential(G_const_ * m_n, dr, r_s);
        }
      }
      // determine target internal energy based on fixed Mach profile
      Real cs_sqr_target  = -sum_phi * inv_Mach_sqr_;
      cs_sqr_target       = Kokkos::max(cs_sqr_target, cs_sqr_floor_);

      // compute next kinetic and internal energy using intermediate cons values
      const Real rho_cons   = cons(mb_id, IDN, k, j, i);
      const Real p_cons_sqr = SQR(cons(mb_id, IM1, k, j, i)) + SQR(cons(mb_id, IM2, k, j, i)) + SQR(cons(mb_id, IM3, k, j, i));
      const Real K_next     = 0.5 * p_cons_sqr / rho_cons;
      const Real U_next     = cs_sqr_target * rho_cons * inv_gm1;

      // set energy
      cons(mb_id, IEN, k, j, i) = U_next + K_next;
    }); // end par_for
    
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::NBodyBlackBodySrcTerm
//! \brief Remove thermal energy from each cell according to black body cooling. Only 
//! valid for 2D simulations, requires physical units to be set in <nbody> athinput block.
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
  const Real gm1 = pmy_pack->phydro->peos->eos_data.gamma - 1.0;

  par_for("nbody_blackbody_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // in 2D, all cells have fixed vertical size
      const Real two_H = size.h_view(mb_id).dx3;
      const Real one_over_4H = 0.5 / two_H;

      // compute cooling rate based on local density and temperature
      const Real tau = prim(mb_id, IDN, k, j, i) * two_H * kappa_es_code_;
      const Real tau_eff = 0.375 * tau + sqrt_3_over_4 + 0.25 / tau;
      const Real T_mid = prim(mb_id, IEN, k, j, i) * gm1 / (prim(mb_id, IDN, k, j, i) * gas_const_code_);
      const Real T_eff = T_mid / tau_eff;
      const Real Q_cool = one_over_4H * rad_const_code_ * c_light_code_ * Kokkos::pow(T_eff, 4); 

      cons(mb_id, IEN, k, j, i) -= Q_cool * beta_dt;
    }); // end par_for
    
  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::NBodyBetaCoolSrcTerm
//! \brief Relax temperature to fixed Mach number profile on either a dynamical or thermal
//! timescale (for BETA_COOL_DYN or BETA_COOL_THERM respectively). Use updated conservative
//! state to step to next, accounting for previous effects of other source terms
void NBody::NBodyBetaCoolSrcTerm(const Real beta_dt) {

  // MeshBlock properties
  auto &indcs    = pmy_pack->pmesh->mb_indcs;
  auto &size     = pmy_pack->pmb->mb_size;
  const int is   = indcs.is;
  const int ie   = indcs.ie;
  const int js   = indcs.js;
  const int je   = indcs.je;
  const int ks   = indcs.ks;
  const int ke   = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &cons     = pmy_pack->phydro->u0;
  auto &prim     = pmy_pack->phydro->w0;

  // privatise nbody data for par_for
  auto &nbody_data_         = nbody_data;
  const int num_nbody_      = num_nbody;
  const Real inv_gm1        = 1.0 / (pmy_pack->phydro->peos->eos_data.gamma - 1.0);
  const Real inv_Mach_sqr_  = inv_Mach_sqr;
  const Real cs_sqr_floor_  = cs_sqr_floor;
  const Real alpha_         = (alpha != 0.0) ? alpha : 0.1;                   // use fiducial value of alpha = 0.1 if not set 
  const Real inv_t_cool_fac = (cooling_mode == BETA_COOL_DYN) ? 1.0 : alpha_; // toggle between dynamical/thermal cooling timescale
  const Real G_const_       = G_const;
  const int soft_mode_      = soft_mode;
  const int inv_2pi         = 1.0 / (2 * M_PI);

  par_for("nbody_beta_cool_src", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {

      // identify cell position
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // compute local sound speed via linear sum of softened potential from each body
      // compute local orbital time via quadratic sum of true orbital frequencies
      Real sum_phi = 0.0;
      Real sum_omega_sqr = 0.0;
      for (int n = 0; n < num_nbody_; n++) {
        // compute true and softened seperation
        const Real dx  = x - nbody_data_.d_view(n, X_DATA);
        const Real dy  = y - nbody_data_.d_view(n, Y_DATA);
        const Real dz  = z - nbody_data_.d_view(n, Z_DATA);
        const Real dr  = Kokkos::sqrt(SQR(dx) + SQR(dy) + SQR(dz)) + FLT_MIN;
        const Real m_n = nbody_data_.d_view(n, M_DATA);
        const Real r_s = nbody_data_.d_view(n, R_SOFT_DATA);
        // compute softened gravitational potential
        if (soft_mode_ == 0) {
          sum_phi += CalcPlummerPotential(G_const_ * m_n, dr, r_s);
        } else {
          sum_phi += CalcSplinePotential(G_const_ * m_n, dr, r_s);
        }
        // compute true orbital frequency
        sum_omega_sqr += G_const_ * m_n * Kokkos::pow(dr, -3.0);
      }
      // determine target internal energy based on fixed Mach profile
      Real cs_sqr_target  = -sum_phi * inv_Mach_sqr_;
      cs_sqr_target       = Kokkos::max(cs_sqr_target, cs_sqr_floor_);
      const Real U_target = cs_sqr_target * prim(mb_id, IDN, k, j, i) * inv_gm1;

      // decay to target internal energy on dynamic or thermal time (set by t_cool_fac)
      const Real inv_t_dyn  = inv_2pi * Kokkos::sqrt(sum_omega_sqr);
      const Real inv_t_cool = inv_t_dyn * inv_t_cool_fac;
      const Real dU         = (prim(mb_id, IEN, k, j, i) - U_target) * (Kokkos::exp(-beta_dt * inv_t_cool) - 1.0);

      // update total energy (dE_kin = 0)
      cons(mb_id, IEN, k, j, i) += dU;
    }); // end par_for
    
  return;
} 

//----------------------------------------------------------------------------------------
//! \fn  void NBody::CalcViscosity
//! \brief Compute the kinematic viscosity in each cell (including ghosts) according to a 
//! constant alpha prescription. Local sound speed informed by the true hydrodynamic value
//! or the ideal value informed by a fixed Mach number (visc mode = 0, 1). Visocisty values
//! are stashed in the nu_iso register for use in NBody::AddViscousFlux which runs after.
void NBody::CalcViscosity(const DvceArray5D<Real> &w0) {

  // MeshBlock properties
  auto &indcs    = pmy_pack->pmesh->mb_indcs;
  auto &size     = pmy_pack->pmb->mb_size;
  // determine par_for limits, including ghosts for active axes
  const int is   = indcs.is - indcs.ng;
  const int ie   = indcs.ie + indcs.ng; 
  const int js   = (pmy_pack->pmesh->multi_d) ? indcs.js - indcs.ng: indcs.js;
  const int je   = (pmy_pack->pmesh->multi_d) ? indcs.je + indcs.ng: indcs.je;
  const int ks   = (pmy_pack->pmesh->three_d) ? indcs.ks - indcs.ng: indcs.ks;
  const int ke   = (pmy_pack->pmesh->three_d) ? indcs.ke + indcs.ng: indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;

  // set register tracking maximal viscosity to zero
  Kokkos::deep_copy(max_nu_iso.view_host(), 0.0);
  max_nu_iso.template modify<HostMemSpace>();
  max_nu_iso.template sync<DevExeSpace>();

  // privatise nbody data for par_for
  auto &nbody_data_        = nbody_data;
  auto &nu_iso_            = nu_iso;
  auto &max_nu_iso_        = max_nu_iso;
  const int num_nbody_     = num_nbody;
  const Real alpha_        = alpha;
  const Real G_const_      = G_const;
  const Real nu_max_       = nu_max;
  const Real visc_mode_    = visc_mode;
  const Real soft_mode_    = visc_mode;
  const Real gm1           = pmy_pack->phydro->peos->eos_data.gamma - 1.0;

  par_for("nbody_calc_visc", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int mb_id, const int k, const int j, const int i) 
    {
      // identify cell position
      const Real x = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(mb_id).x1min, size.d_view(mb_id).x1max);
      const Real y = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(mb_id).x2min, size.d_view(mb_id).x2max);
      const Real z = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(mb_id).x3min, size.d_view(mb_id).x3max);

      // compute local omega via quadratic sum of true orbital frequencies
      Real sum_omega_sqr = 0.0;
      for (int n = 0; n < num_nbody_; n++) {
        // compute true and softened seperation
        const Real dx = x - nbody_data_.d_view(n, X_DATA);
        const Real dy = y - nbody_data_.d_view(n, Y_DATA);
        const Real dz = z - nbody_data_.d_view(n, Z_DATA);
        const Real dr = Kokkos::sqrt(SQR(dx) + SQR(dy) + SQR(dz)) + FLT_MIN;
        // compute orbital frequency
        sum_omega_sqr += G_const_ * nbody_data_.d_view(n, M_DATA) * Kokkos::pow(dr, -3.0);
      }

      // calculate local sound speed from primitive state
      const Real cs_sqr = w0(mb_id, IEN, k, j, i) * gm1 / w0(mb_id, IDN, k, j, i); 

      // compute local kinematic viscosity using alpha prescripton
      Real nu = alpha_ * cs_sqr * Kokkos::pow(sum_omega_sqr, -0.5);
      nu = Kokkos::min(nu, nu_max_);

      // stash viscosity state in register
      nu_iso_(mb_id, k, j, i) = nu;

      // thread-safe maximum check for tracker
      Kokkos::atomic_max(&max_nu_iso_.d_view(mb_id), nu);
    }); // end par_for
    
  // enforce update of maximum tracker on host
  max_nu_iso.template modify<DevExeSpace>();
  max_nu_iso.template sync<HostMemSpace>();

  return;
}

//----------------------------------------------------------------------------------------
//! \fn  void NBody::CalcViscosity
//! \brief Called after NBody::CalcViscosity has populated the nu_iso array with the 
//! per-cell kinematic viscosity. Computes the face-fluxes driven by viscosity and updates 
//! the conserved fluxes
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

//----------------------------------------------------------------------------------------
//! \fn  void NBody::NewViscousTimeStep
//! \brief Computes the maximum timestep for stable viscous forward integration, based on
//! the nu_iso array populated by NBody::CalcViscosity
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

//----------------------------------------------------------------------------------------
//! \fn  void NBodyHistory
//! \brief Packages information on the current nbody_data state to write as history output
//! Defined as non-member function to allow assignment pre-NBody construction in pgen.
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
    std::string n_str = std::to_string(n);
    pdata->label[M_HIST + hist_offset]       = "m" + n_str;
    pdata->label[X_HIST + hist_offset]       = "x" + n_str;
    pdata->label[Y_HIST + hist_offset]       = "y" + n_str;
    pdata->label[Z_HIST + hist_offset]       = "z" + n_str;
    pdata->label[VX_HIST + hist_offset]      = "vx" + n_str;
    pdata->label[VY_HIST + hist_offset]      = "vy" + n_str;
    pdata->label[VZ_HIST + hist_offset]      = "vz" + n_str;
    pdata->label[MDOT_HIST + hist_offset]    = "mdot" + n_str;
    pdata->label[AX_GRAV_HIST + hist_offset] = "ax_grav" + n_str;
    pdata->label[AY_GRAV_HIST + hist_offset] = "ay_grav" + n_str;
    pdata->label[AZ_GRAV_HIST + hist_offset] = "az_grav" + n_str;
    pdata->label[AX_ACC_HIST + hist_offset]  = "ax_acc" + n_str;
    pdata->label[AY_ACC_HIST + hist_offset]  = "ay_acc" + n_str;
    pdata->label[AZ_ACC_HIST + hist_offset]  = "az_acc" + n_str;
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

//----------------------------------------------------------------------------------------
//! \fn  void NBodyTrackRefinementCondition
//! \brief Determine the refinement flags for all MeshBlocks in pack, based on proximity
//! to bodies within the domain, and their respective radii/levels. Defined as non-member
//! function to allow for assignment pre-NBody construction in pgen.
void NBodyTrackRefinementCondition(MeshBlockPack* pmbp) {
  
  // do not attempt write if no NBody instance
  if (pmbp->pnbody == nullptr) return; 

  auto &refine_flag = pmbp->pmesh->pmr->refine_flag;
  int mbs = pmbp->pmesh->gids_eachrank[global_variable::my_rank];
  int nmb = pmbp->nmb_thispack;
  auto &size = pmbp->pmb->mb_size;
  auto &multi_d = pmbp->pmesh->multi_d;
  auto &three_d = pmbp->pmesh->three_d;
  Real time = pmbp->pmesh->time;

  // loop over MeshBlocks in this MeshBlockPack
  // MeshBlock count small, perfom on host
  for (int mb_id = 0; mb_id < nmb; mb_id++) {

    // by default, mark for derefinement (level decrease)
    int mb_refine_flag = -1;

    // extract MeshBlock bounds
    Real &x1min = size.h_view(mb_id).x1min;
    Real &x1max = size.h_view(mb_id).x1max;
    Real &x2min = size.h_view(mb_id).x2min;
    Real &x2max = size.h_view(mb_id).x2max;
    Real &x3min = size.h_view(mb_id).x3min;
    Real &x3max = size.h_view(mb_id).x3max;

    // cycle over bodies
    for (int n = 0; n < pmbp->pnbody->num_nbody; n++) {
      
      // if refinement not live for body n, skip
      if (time < pmbp->pnbody->nbody_data.h_view(n, T_AMR_DATA)) continue;
      
      // if refinement radius for body is zero, skip
      const Real rad = pmbp->pnbody->nbody_data.h_view(n, R_AMR_DATA);
      if (rad == 0.0) continue;

      // read body position 
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
            // overlap confirmed, check if at maximal level for this body
            const int this_mb_refinement = pmbp->pmesh->lloc_eachmb[mb_id+mbs].level - pmbp->pmesh->root_level;
            const int target_refinement  = pmbp->pnbody->nbody_data.h_view(n, N_AMR_DATA);
            if (this_mb_refinement < target_refinement) { // mark for level increase
              mb_refine_flag = 1;
              break; // if marked for refinement, skip checking other bodies
            } else if (this_mb_refinement == target_refinement) { // mark for fixed level
              mb_refine_flag = 0;
              continue; // continue check, other body may motivate level increase
            } // refinement flag check
          } // three_d check
        } // multi_d check
      } // x1 check
    } // end n loop

    // stash refinement flag
    refine_flag.h_view(mb_id + mbs) = mb_refine_flag;
  } // end mb_id loop

  // update refine_flag on device
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn  Real CalcSplineAcceleration
//! \brief Compute the acceleration due to a gravitating body, according to the a softened
//! potential on a cubic spline. Portable to __host__ and __device__
KOKKOS_FUNCTION Real CalcSplineAcceleration(const Real Gm, const Real r, const Real h) {
  
  // check if in the Newtonian far field
  if (r >= h) { 
    return -Gm / SQR(r);
  } 
  
  // use forcing softened on spline
  const Real g_fac = Gm / SQR(h);
  const Real d = r / h;
  if (r > 0.5 * h) { // mid field, use softened
    const Real spline = 0.066667 / SQR(d)
                      + d * (-21.33333 + d * (48 + d * (-38.4 + d * 10.66667)));
                      // 1/(15d^2) - 64/3 d + 48d^2 - 192/5 d^3 + 32/3 d^4
    return g_fac * spline;
  } else { // near field
    const Real spline = d * (-10.66667 + d * d * (38.4 - 32 * d)); 
                      // -32/3 d + 192/5 d^3 - 32d^4
    return g_fac * spline;
  }
}

//----------------------------------------------------------------------------------------
//! \fn  Real CalcSplinePotential
//! \brief Compute the potential due to a gravitating body, according to the a softened
//! potential on a cubic spline. Portable to __host__ and __device__
KOKKOS_FUNCTION Real CalcSplinePotential(const Real Gm, const Real r, const Real h) {

  // check if in the Newtonian far field
  if (r >= h) {
    return -Gm / r;
  }

  // use potential softened on spline
  const Real phi_fac = -Gm / h;
  const Real d = r / h;
  if (r > 0.5 * h) { // mid field
    const Real spline = -0.066667 / d + 3.2 
                        + SQR(d) * (-10.6667 + d * (16 + d * (-9.6 + d * 2.1333)));
                        // -1/(15d) - 16/5 - 32/3 d^2 + 16d^3 - 48/5 d^4 + 32/15 d^5
    return phi_fac * spline;
  } else { // near field
    const Real spline = 2.8 + SQR(d) * (-5.3333 + SQR(d) * (9.6 - 6.4 * d));
                        // 14/5 - 16/3 d^2 + 48/5 d^4 - 32/5d^4
    return phi_fac * spline;
  }  
}

//----------------------------------------------------------------------------------------
//! \fn  Real CalcPlummerAcceleration
//! \brief Compute the acceleration due to a gravitating body, according to the a softened
//! plummer potential. Portable to __host__ and __device__
KOKKOS_FUNCTION Real CalcPlummerAcceleration(const Real Gm, const Real r, const Real h) {
  
  return -Gm * r * Kokkos::pow(SQR(r) + SQR(h), -1.5);
}

//----------------------------------------------------------------------------------------
//! \fn  Real CalcPlummerPotential
//! \brief Compute the potential due to a gravitating body, according to the a softened
//! plummer potential. Portable to __host__ and __device__
KOKKOS_FUNCTION Real CalcPlummerPotential(const Real Gm, const Real r, const Real h) {

  return -Gm * Kokkos::pow(SQR(r) + SQR(h), -0.5);
}

} // end nbody namespace


