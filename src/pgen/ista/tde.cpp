// this problem generator is designed to be used to study the minidisc TDE problem
// it inherits much of its structure from the cbdiso_3d.cpp pgen
// pgen is UNDER CONSTRUCTION and untested

#include <math.h>
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <Kokkos_MathematicalFunctions.hpp>

#include "parameter_input.hpp"
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "coordinates/cell_locations.hpp"

#include "nbody/nbody.hpp"
#include "globals.hpp"

// nbody problem generator
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  // load nbody properties
  int num_nbody = pin->GetOrAddInteger("nbody", "num_nbody", 0);
  bool hist_nbody = pin->GetOrAddBoolean("nbody", "hist_nbody", false);
  if (hist_nbody) {
    user_hist = true;
    user_hist_func = nbody::NBodyHistory; 
  }

  // enroll AMR if flagged
  user_ref_func = nbody::NBodyTrackRefinementCondition;

  // Skip initialization if this is a restart
  if (restart) return;

  // Initial conditions. (1) Get mesh (which is in scope of pgen class)
  auto &indcs = pmy_mesh_->mb_indcs;
  int &is     = indcs.is; 
  int &js     = indcs.js; 
  int &ks     = indcs.ks; 
  const int is_wg   = indcs.is - indcs.ng;
  const int ie_wg   = indcs.ie + indcs.ng; 
  const int js_wg   = (pmy_mesh_->multi_d) ? indcs.js - indcs.ng: indcs.js;
  const int je_wg   = (pmy_mesh_->multi_d) ? indcs.je + indcs.ng: indcs.je;
  const int ks_wg   = (pmy_mesh_->three_d) ? indcs.ks - indcs.ng: indcs.ks;
  const int ke_wg   = (pmy_mesh_->three_d) ? indcs.ke + indcs.ng: indcs.ke;
  
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size          = pmbp->pmb->mb_size;
  const bool three_d  = pmy_mesh_->three_d;

  // load minidisc properties
  const int prograde    = pin->GetOrAddBoolean("problem", "prograde", true);
  const Real r_minidisc = pin->GetOrAddReal("problem", "r_minidisc", 0.25);
  const Real r_cavity   = pin->GetOrAddReal("problem", "r_cavity", 0.05);
  const Real rho0       = pin->GetOrAddReal("problem", "rho0", 1.0);
  
  // thermodynamic properties
  bool is_ideal    = (pin->GetOrAddString("hydro", "eos", "ideal") == "ideal");
  const Real Mach  = pin->GetOrAddReal("nbody", "Mach", 10.0);
  const Real alpha = pin->GetOrAddReal("nbody", "alpha", 0.0);

  // package body properties
  DualArray2D<Real> pgen_nbody_data;
  int pgen_nvar_data = 11; // number of variables in pgen_nbody_data register
  Kokkos::realloc(pgen_nbody_data, num_nbody, pgen_nvar_data); // NBodyDataIndices

  for (int n = 0; n < num_nbody; n++) {
    std::string str_n = std::to_string(n);
    // standard register imports, as in NBody
    pgen_nbody_data.h_view(n, 0) = pin->GetReal("nbody", "m" + str_n);
    pgen_nbody_data.h_view(n, 1) = pin->GetReal("nbody", "x" + str_n);
    pgen_nbody_data.h_view(n, 2) = pin->GetReal("nbody", "y" + str_n);
    pgen_nbody_data.h_view(n, 3) = pin->GetReal("nbody", "z" + str_n);
    pgen_nbody_data.h_view(n, 4) = pin->GetReal("nbody", "vx" + str_n);
    pgen_nbody_data.h_view(n, 5) = pin->GetReal("nbody", "vy" + str_n);
    pgen_nbody_data.h_view(n, 6) = pin->GetReal("nbody", "vz" + str_n);
    pgen_nbody_data.h_view(n, 7) = pin->GetReal("nbody", "r_soft" + str_n);
    // initial conditions imports
    pgen_nbody_data.h_view(n, 8) = pin->GetReal("nbody", "grade" + str_n);
    pgen_nbody_data.h_view(n, 9) = pin->GetReal("nbody", "r_cav" + str_n);
    pgen_nbody_data.h_view(n, 10) = pin->GetReal("nbody", "r_disk" + str_n);
  } // end nbody loop

  // push properties to device
  pgen_nbody_data.template modify<HostMemSpace>();
  pgen_nbody_data.template sync<DevExeSpace>();

  // determine build mode (single or double for minidiscs)
  int host_index = -1; // initially unformatted
  for (int n = 0; n < num_nbody; n++) {
    if (pgen_nbody_data.h_view(n, 10) > 0.0) {
      if (host_index != -1) {
        host_index = n; // first valid host identified
      } else {
        host_index = -2; // multiple hosts identified, build multiple minidiscs
      }
    }
  } // end nbody loop

  // set initial hydrodynamic state
  if (pmbp->phydro != nullptr) 
  {

    auto &w0_                = pmbp->phydro->w0;  // Primitive variables (density, velocity, pressure)
    const Real inv_Mach      = 1.0 / Mach;
    const Real inv_Mach_sqr  = SQR(inv_Mach);
    const Real inv_gm1       = 1.0 / (pmbp->phydro->peos->eos_data.gamma - 1.0);
    const Real cs_sqr_iso    = pin->GetOrAddReal("hydro", "iso_sound_speed", 1.0);
    const Real rho_floor_fac = pin->GetOrAddReal("problem", "rho_floor_fac", 1e-8);
    const Real cs_sqr_floor  = pin->GetOrAddReal("problem", "cs_sqr_floor", 1e-8);
    const Real G_const       = pin->GetOrAddReal("nbody", "G_const", 1.0);
    const int soft_mode      = pin->GetOrAddInteger("nbody", "soft_mode", 0); 

    // (3) loop over cells
    par_for("pgen_tde", 
            DevExeSpace(),               // CPU or GPU execution space
            0, (pmbp->nmb_thispack-1),   // Loop 1: m = mesh block index (0 to N-1)
            ks_wg, ke_wg,                      // Loop 2: k = z index
            js_wg, je_wg,                      // Loop 3: j = y index
            is_wg, ie_wg,                      // Loop 4: i = x index
    
    KOKKOS_LAMBDA(int m, int k, int j, int i) { 

      // update coordinates for MeshBlock m
      Real &x1min = size.d_view(m).x1min;                   // xmin
      Real &x1max = size.d_view(m).x1max;                   // xmax
      int nx1     = indcs.nx1;                              // nx
      Real x1v    = CellCenterX(i-is, nx1, x1min, x1max);   // x coordinate

      Real &x2min = size.d_view(m).x2min;                   // ymin
      Real &x2max = size.d_view(m).x2max;                   // ymax
      int nx2     = indcs.nx2;                              // ny
      Real x2v    = CellCenterX(j-js, nx2, x2min, x2max);   // y coordinate

      Real &x3min = size.d_view(m).x3min;                   // zmin
      Real &x3max = size.d_view(m).x3max;                   // zmax
      int nx3     = indcs.nx3;                              // nz
      Real x3v    = CellCenterX(k-ks, nx3, x3min, x3max);   // z coordinate

      // construct IC by build configuration
      if (host_index == -2) {
        // build multiple minidiscs

        // iterate over potential hosts, determine contributions
        // density: linear sum of minidisc contributions (assuming no overlap)
        // velocity: weighted by local angular frequency
        // sound speed: determine by constant Mach profile
        Real sum_rho         = 0.0;
        Real sum_phi         = 0.0;
        Real sum_omega_sqr   = 0.0;
        Real weighted_vx_sum = 0.0; 
        Real weighted_vy_sum = 0.0; 
        Real weighted_vz_sum = 0.0;
        for (int n = 0; n < num_nbody; n++) {
          // unpack host properties
          const Real m_host      = pgen_nbody_data.d_view(n, 0);
          const Real x_host      = pgen_nbody_data.d_view(n, 1);
          const Real y_host      = pgen_nbody_data.d_view(n, 2);
          const Real z_host      = pgen_nbody_data.d_view(n, 3);
          const Real vx_host     = pgen_nbody_data.d_view(n, 4);
          const Real vy_host     = pgen_nbody_data.d_view(n, 5);
          const Real vz_host     = pgen_nbody_data.d_view(n, 6);
          const Real r_soft_host = pgen_nbody_data.d_view(n, 7);
          const Real grade_host  = pgen_nbody_data.d_view(n, 8);
          const Real r_cav_host  = pgen_nbody_data.d_view(n, 9);
          const Real r_disk_host = pgen_nbody_data.d_view(n, 10);

          // if host has no minidisc, skip to next body
          if (r_disk_host <= 0.0) continue;

          // determine distance to host
          const Real R_sqr = SQR(x1v - x_host) + SQR(x2v - y_host);
          const Real z = x3v - z_host;
          const Real R = Kokkos::sqrt(R_sqr);

          // set density using radial and vertical profile
          // disc is ALWAYS in the x-y plane, orbit is rotated
          const Real outer_profile = 1.0 / Kokkos::cosh(Kokkos::pow(R/r_disk_host,4.0));
          Real radial_profile;
          if (alpha == 0.0) {
            // constant nu -> constant surface density
            radial_profile = 1.0;
          } else {
            // constant alpha -> surface density propto r^-0.5
            if (R < r_soft_host) { // set as flat within sink
              radial_profile = Kokkos::pow(r_soft_host / r_disk_host, -0.5);
            } else {
              radial_profile = Kokkos::pow(R / r_disk_host, -0.5);
            }
          }
          const Real inner_profile = rho_floor_fac + (1.0 - rho_floor_fac) * Kokkos::exp(-Kokkos::pow((r_cav_host / R), 4.0));
          const Real H_sqr = R_sqr * inv_Mach_sqr;
          const Real vertical_profile = Kokkos::exp(-0.5 * SQR(z) / H_sqr); // unity if 2D
          Real rho = rho0 * radial_profile * inner_profile * outer_profile * vertical_profile;

          // set velocity in disc (everything softened-Keplerian about host)
          // velocity profile is such that g + v^2 / r + dPdr / rho = 0
          Real g_soft;
          if (soft_mode == 0) {
            g_soft = nbody::CalcPlummerAcceleration(G_const * m_host, R, r_soft_host);
          } else {
            g_soft = nbody::CalcSplineAcceleration(G_const * m_host, R, r_soft_host);
          }
          Real v_phi = Kokkos::sqrt(-g_soft * R * (1.0 - inv_Mach_sqr));
          if (grade_host != 1.0) { // if retrograde, flip sign of velocity
            v_phi *= -1.0;
          }
          const Real phi = Kokkos::atan2(x2v - y_host, x1v - x_host); // RH argument from +x axis
          const Real vx = vx_host - v_phi * Kokkos::sin(phi);
          const Real vy = vy_host + v_phi * Kokkos::cos(phi);
          const Real vz = vz_host;

          // add local potential to total
          Real phi_grav;
          if (soft_mode == 0) {
            phi_grav = nbody::CalcPlummerPotential(G_const * m_host, R, r_soft_host);
          } else {
            phi_grav = nbody::CalcSplinePotential(G_const * m_host, R, r_soft_host);
          }

          // compute local orbital frequnecy
          const Real omega_sqr = G_const * pgen_nbody_data.d_view(n, 0) * Kokkos::pow(R, -3.0);
          
          // add local values to sum
          sum_omega_sqr   += omega_sqr;       // used to determine weighting
          sum_rho         += rho;             // linear sum (no minidisc overlap)
          weighted_vx_sum += vx * omega_sqr;  // weighted by square orbital frequency
          weighted_vy_sum += vy * omega_sqr;
          weighted_vz_sum += vz * omega_sqr;
          sum_phi         += phi_grav;        // coherant with constant Mach number profile
        } // end nbody loop

        // compute local hydrodynamic properties (applying normalisation and floors)
        const Real rho  = Kokkos::max(sum_rho, rho0 * rho_floor_fac);
        const Real norm = 1.0 / sum_omega_sqr;
        const Real vx   = weighted_vx_sum * norm;
        const Real vy   = weighted_vy_sum * norm;
        const Real vz   = weighted_vz_sum * norm;
        Real cs_sqr     = -sum_phi * inv_Mach_sqr;
        cs_sqr          = Kokkos::max(cs_sqr, cs_sqr_floor);
        const Real U    = cs_sqr * rho * inv_gm1;

        // stash primitive values
        w0_(m, IDN, k, j, i)               = rho;              
        w0_(m, IVX, k, j, i)               = vx;               
        w0_(m, IVY, k, j, i)               = vy;               
        if (three_d) w0_(m, IVZ, k, j, i)  = vz;             
        if (is_ideal) w0_(m, IEN, k, j, i) = U;   
      } else {
        // build single minidisc about body indexed by host_index

        // unpack host properties
        const Real m_host = pgen_nbody_data.d_view(host_index, 0);
        const Real x_host = pgen_nbody_data.d_view(host_index, 1);
        const Real y_host = pgen_nbody_data.d_view(host_index, 2);
        const Real z_host = pgen_nbody_data.d_view(host_index, 3);
        const Real vx_host = pgen_nbody_data.d_view(host_index, 4);
        const Real vy_host = pgen_nbody_data.d_view (host_index, 5);
        const Real vz_host = pgen_nbody_data.d_view(host_index, 6);
        const Real r_soft_host = pgen_nbody_data.d_view(host_index, 7);
        const Real grade_host = pgen_nbody_data.d_view(host_index, 8);
        const Real r_cav_host = pgen_nbody_data.d_view(host_index, 9);
        const Real r_disk_host = pgen_nbody_data.d_view(host_index, 10);

        // determine distance to host
        const Real R_sqr = SQR(x1v - x_host) + SQR(x2v - y_host);
        const Real z = x3v - z_host;
        const Real R = Kokkos::sqrt(R_sqr);

        // set density using radial and vertical profile
        // disc is ALWAYS in the x-y plane, orbit is rotated
        const Real outer_profile = 1.0 / Kokkos::cosh(Kokkos::pow(R/r_disk_host,4.0));
        Real radial_profile;
        if (alpha == 0.0) {
          // constant nu -> constant surface density
          radial_profile = 1.0;
        } else {
          // constant alpha -> surface density propto r^-0.5
          if (R < r_soft_host) { // set as flat within sink
            radial_profile = Kokkos::pow(r_soft_host / r_disk_host, -0.5);
          } else {
            radial_profile = Kokkos::pow(R / r_disk_host, -0.5);
          }
        }
        const Real inner_profile = rho_floor_fac + (1.0 - rho_floor_fac) * Kokkos::exp(-Kokkos::pow((r_cav_host / R), 4.0));
        const Real H_sqr = R_sqr * inv_Mach_sqr;
        const Real vertical_profile = Kokkos::exp(-0.5 * SQR(z) / H_sqr); // unity if 2D
        Real rho = rho0 * radial_profile * inner_profile * outer_profile * vertical_profile;
        rho = Kokkos::max(rho, rho0 * rho_floor_fac);

        // set velocity in disc (everything softened-Keplerian about host)
        // velocity profile is such that g + v^2 / r + dPdr / rho = 0
        Real g_soft;
        if (soft_mode == 0) {
          g_soft = nbody::CalcPlummerAcceleration(G_const * m_host, R, r_soft_host);
        } else {
          g_soft = nbody::CalcSplineAcceleration(G_const * m_host, R, r_soft_host);
        }
        Real v_phi = Kokkos::sqrt(-g_soft * R * (1.0 - inv_Mach_sqr));
        if (grade_host != 1.0) { // if retrograde, flip sign of velocity
            v_phi *= -1.0;
          }
        const Real phi = Kokkos::atan2(x2v - y_host, x1v - x_host); // RH argument from +x axis
        const Real vx = vx_host - v_phi * Kokkos::sin(phi);
        const Real vy = vy_host + v_phi * Kokkos::cos(phi);
        const Real vz = vz_host;

        // set pressure using fixed Mach number prescription
        Real phi_grav;
        if (soft_mode == 0) {
          phi_grav = nbody::CalcPlummerPotential(G_const * m_host, R, r_soft_host);
        } else {
          phi_grav = nbody::CalcSplinePotential(G_const * m_host, R, r_soft_host);
        }
        Real cs_sqr_local = -phi_grav * inv_Mach_sqr;
        cs_sqr_local      = Kokkos::max(cs_sqr_local, cs_sqr_floor);
        const Real U      = cs_sqr_local * rho * inv_gm1;
        
        // ===== Set primitive variables =====
        w0_(m, IDN, k, j, i)               = rho;              
        w0_(m, IVX, k, j, i)               = vx;               
        w0_(m, IVY, k, j, i)               = vy;               
        if (three_d) w0_(m, IVZ, k, j, i)  = vz;             
        if (is_ideal) w0_(m, IEN, k, j, i) = U;   
      } // end single minidisc build
    }); // end parfor

    // ===== Convert primitives to conserved variables =====
    pmbp->phydro->peos->PrimToCons(w0_, pmbp->phydro->u0, is_wg, ie_wg, js_wg, je_wg, ks_wg, ke_wg);
  } // end hydro init 
  std::cout << "Completed ProblemGenerator::UserProblem on rank " << global_variable::my_rank << std::endl;
} 

// ======================== User-Defined Source Terms =========================
// ============================================================================

