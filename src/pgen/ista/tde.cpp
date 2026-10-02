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
  const Real Mach = pin->GetOrAddReal("problem", "Mach", 10.0);
  bool is_ideal = (pin->GetOrAddString("hydro", "eos", "ideal") == "ideal");
  const Real alpha = pin->GetOrAddReal("problem", "alpha", 0.0);

  // isolate initial binary state (NBody not assured init during pgen)
  const Real m_prim = pin->GetReal("nbody", "m0");
  const Real x_prim = pin->GetReal("nbody", "x0");
  const Real y_prim = pin->GetReal("nbody", "y0");
  const Real z_prim = pin->GetReal("nbody", "z0");
  const Real vx_prim = pin->GetReal("nbody", "vx0");
  const Real vy_prim = pin->GetReal("nbody", "vy0");
  const Real vz_prim = pin->GetReal("nbody", "vz0");
  const Real r_soft_prim = pin->GetReal("nbody", "r_soft0");

  const Real m_sec = pin->GetReal("nbody", "m1");
  const Real x_sec = pin->GetReal("nbody", "x1");
  const Real y_sec = pin->GetReal("nbody", "y1");
  const Real z_sec = pin->GetReal("nbody", "z1");
  const Real vx_sec = pin->GetReal("nbody", "vx1");
  const Real vy_sec = pin->GetReal("nbody", "vy1");
  const Real vz_sec = pin->GetReal("nbody", "vz1");

  // (2) access prims from mesh block pack
  if (pmbp->phydro != nullptr) 
  {

    auto &w0_ = pmbp->phydro->w0;  // Primitive variables (density, velocity, pressure)
    const Real inv_Mach = 1.0 / Mach;
    const Real inv_gm1 = 1.0 / (pmbp->phydro->peos->eos_data.gamma - 1.0);
    const Real cs_sqr_floor = 1e-8;
    Real cs_sqr = pin->GetOrAddReal("hydro", "iso_sound_speed", 1.0);

    // (3) loop over cells
    par_for("pgen_tde", 
            DevExeSpace(),               // CPU or GPU execution space
            0, (pmbp->nmb_thispack-1),   // Loop 1: m = mesh block index (0 to N-1)
            ks, ke,                      // Loop 2: k = z index
            js, je,                      // Loop 3: j = y index
            is, ie,                      // Loop 4: i = x index
    
    KOKKOS_LAMBDA(int m, int k, int j, int i) { 
    // Lambda function for every (m,k,j,i) combination:

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

      // determine distance from primary
      const Real r_sqr = SQR(x1v - x_prim) + SQR(x2v - y_prim) + SQR(x3v - z_prim);
      const Real r = Kokkos::sqrt(r_sqr);
      const Real soft_r_sqr = SQR(r_soft_prim) + r_sqr;

      // set density using inverse cavity kernel
      const Real delta_floor = 1e-6;
      //const Real cavity_func = (1.0 - delta_floor) * Kokkos::exp(-Kokkos::pow((r_cavity / r), 12.0)); 
      const Real cavity_func = 1.0; // TEMP run without cavity for low-res runs
      const Real disc_func = 1.0 - Kokkos::exp(-Kokkos::pow((r_minidisc / (r - r_minidisc)), 12.0));
      const Real rho = rho0 * (disc_func * cavity_func + delta_floor);

      // set velocity in disc (everything Keplerian about primary, including cavity)
      // velocity profile should match softened potential
      const Real v_phi_sqr = m_prim * r / soft_r_sqr;
      const Real v_phi = Kokkos::sqrt(v_phi_sqr);
      if (prograde) {
        v_phi = v_phi;
      } else {
        v_phi *= -1.0;
      }
      const Real phi = Kokkos::atan2(x2v - y_prim, x1v - x_prim); // RH argument from +x axis
      const Real vx = vx_prim - v_phi * Kokkos::sin(phi);
      const Real vy = vy_prim + v_phi * Kokkos::cos(phi);
      const Real vz = 0.0;
      // ^ todo: generalise this to 3D

      // set pressure
      const Real cs_sqr_local = SQR(v_phi / Mach) + cs_sqr_floor;
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

