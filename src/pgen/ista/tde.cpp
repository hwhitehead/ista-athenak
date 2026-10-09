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
  int &is     = indcs.is; int &ie = indcs.ie;
  int &js     = indcs.js; int &je = indcs.je;
  int &ks     = indcs.ks; int &ke = indcs.ke;
  
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  auto &size          = pmbp->pmb->mb_size;
  const bool is_3d = pmy_mesh_->three_d;

  // load minidisc properties
  const int prograde = pin->GetOrAddBoolean("problem", "prograde", true);
  const Real r_minidisc = pin->GetOrAddReal("problem", "r_minidisc", 0.25);
  const Real r_cavity = pin->GetOrAddReal("problem", "r_cavity", 0.05);
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real Mach = pin->GetOrAddReal("nbody", "Mach", 10.0);
  bool is_ideal = (pin->GetOrAddString("hydro", "eos", "ideal") == "ideal");
  const Real alpha = pin->GetOrAddReal("problem", "alpha", 0.0);

  // define host, with check for number of bodies
  bool orbit_primary = pin->GetOrAddBoolean("problem", "orbit_primary", true);
  if (num_nbody == 1) {
    // enforce primary as host
    orbit_primary = true;
  } else if (num_nbody == 0){
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in ProblemGenerator::UserProblem" << std::endl
                << "Require num_nbody > 0 to host disc" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // set host properties
  std::string host_index = (orbit_primary) ? "0" : "1";
  const Real m_host      = pin->GetReal("nbody", "m" + host_index);
  const Real x_host      = pin->GetReal("nbody", "x" + host_index);
  const Real y_host      = pin->GetReal("nbody", "y" + host_index);
  const Real z_host      = pin->GetReal("nbody", "z" + host_index);
  const Real vx_host     = pin->GetReal("nbody", "vx" + host_index);
  const Real vy_host     = pin->GetReal("nbody", "vy" + host_index);
  const Real vz_host     = pin->GetReal("nbody", "vz" + host_index);
  const Real r_soft_host = pin->GetReal("nbody", "r_soft" + host_index);

  // set initial hydrodynamic state
  if (pmbp->phydro != nullptr) 
  {

    auto &w0_ = pmbp->phydro->w0;  // Primitive variables (density, velocity, pressure)
    const Real inv_Mach = 1.0 / Mach;
    const Real inv_Mach_sqr = SQR(inv_Mach);
    const Real inv_gm1 = 1.0 / (pmbp->phydro->peos->eos_data.gamma - 1.0);
    Real cs_sqr = pin->GetOrAddReal("hydro", "iso_sound_speed", 1.0);
    const Real rho_floor_fac = pin->GetOrAddReal("problem", "rho_floor_fac", 1e-8);
    const Real cs_sqr_floor = pin->GetOrAddReal("problem", "cs_sqr_floor", 1e-8);
    const Real G_const = pin->GetOrAddReal("nbody", "G_const", 1.0);

    // (3) loop over cells
    par_for("pgen_tde", 
            DevExeSpace(),               // CPU or GPU execution space
            0, (pmbp->nmb_thispack-1),   // Loop 1: m = mesh block index (0 to N-1)
            ks, ke,                      // Loop 2: k = z index
            js, je,                      // Loop 3: j = y index
            is, ie,                      // Loop 4: i = x index
    
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

      // determine distance from host in cylindrical and spherical polars 
      const Real R_sqr = SQR(x1v - x_host) + SQR(x2v - y_host);
      const Real z = x3v - z_host;
      // const Real r_sqr = R_sqr + SQR(x3v - z_parent);
      const Real R = Kokkos::sqrt(R_sqr);
      const Real soft_R_sqr = SQR(r_soft_host) + R_sqr;

      // set density using radial and vertical profile
      // disc is ALWAYS in the x-y plane, orbit is rotated
      const Real radial_profile = 1.0 / Kokkos::cosh(Kokkos::pow(R/r_minidisc,4.0));
      const Real H_sqr = R_sqr * inv_Mach_sqr;
      const Real vertical_profile = Kokkos::exp(-0.5 * SQR(z) / H_sqr); // unity if 2D
      const Real rho = rho0 * (radial_profile * vertical_profile + rho_floor_fac);

      // set velocity in disc (everything Keplerian about host)
      // velocity profile should match softened potential
      const Real v_phi_sqr = G_const * m_host * R / soft_R_sqr;
      Real v_phi = Kokkos::sqrt(v_phi_sqr);
      if (prograde) {
        v_phi = v_phi;
      } else {
        v_phi *= -1.0;
      }
      const Real phi = Kokkos::atan2(x2v - y_host, x1v - x_host); // RH argument from +x axis
      const Real vx = vx_host - v_phi * Kokkos::sin(phi);
      const Real vy = vy_host + v_phi * Kokkos::cos(phi);
      const Real vz = vz_host;

      // set pressure
      const Real cs_sqr_local = SQR(v_phi) * inv_Mach_sqr + cs_sqr_floor;
      const Real P = cs_sqr_local * rho;
      
      // ===== Set primitive variables =====
      w0_(m, IDN, k, j, i) = rho;              
      w0_(m, IVX, k, j, i) = vx;               
      w0_(m, IVY, k, j, i) = vy;               
      if (is_3d) w0_(m, IVZ, k, j, i) = vz;             
      if (is_ideal) w0_(m, IPR, k, j, i) = P;   
    }); 

    // ===== Convert primitives to conserved variables =====
    pmbp->phydro->peos->PrimToCons(w0_, pmbp->phydro->u0, is, ie, js, je, ks, ke);
  } // end hydro init 
  std::cout << "Completed ProblemGenerator::UserProblem on rank " << global_variable::my_rank << std::endl;
} 

// ======================== User-Defined Source Terms =========================
// ============================================================================

