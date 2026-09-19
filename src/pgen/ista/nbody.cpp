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

// TODO: package this as a standard output format
void NBodyHistory(HistoryData *pdata, Mesh *pm);

// nbody problem generator
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {

  // load nbody properties
  int num_nbody = pin->GetOrAddInteger("nbody", "num_nbody", 0);
  bool hist_nbody = pin->GetOrAddBoolean("nbody", "hist_nbody", false);
  if (hist_nbody) {
    user_hist = true;
    user_hist_func = &NBodyHistory; // TODO: embed into nbody class
  }

  // TODO: check this with david
  // Free the Kokkos::View accumulators before Kokkos::finalize() runs (they have
  // static storage duration, so they'd be destroyed after main() returns).
  // pgen_final_func   = &CircumbinaryFinalAnalysis;

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

  // load disc properties TODO: commmunicate to forced isotherm, new class or stash in nbody
  Real m_sum = 0.0; // sum over bodies
  for (int n = 0; n < num_nbody; n++) {
    const Real m_n = pin->GetOrAddReal("nbody", "m" + std::to_string(n), 0.0);
    m_sum += m_n;
  }
  const Real Gm_sum = m_sum; // assume G = 1
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real Mach = pin->GetOrAddReal("problem", "Mach", 10.0);
  const Real h_sqr = 1.0 / (Mach * Mach);
  const Real r_cavity = pin->GetOrAddReal("problem", "r_cavity", 2.0);
  bool is_ideal = (pin->GetOrAddString("hydro", "eos", "ideal") == "ideal");
  const Real alpha = pin->GetOrAddReal("problem", "alpha", 0.0);

  // (2) access prims from mesh block pack
  if (pmbp->phydro != nullptr) 
  {

    auto &w0_ = pmbp->phydro->w0;  // Primitive variables (density, velocity, pressure)

    // (3) loop over cells
    par_for("pgen_nbody", 
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

      // determine distance from barycenter
      const Real r_sqr = x1v * x1v + x2v * x2v + x3v * x3v;
      const Real r = Kokkos::sqrt(r_sqr);

      // set density by cavity kernel
      const Real cavity_fac = 1e-5 + (1.0 - 1e-5) * Kokkos::exp(-Kokkos::pow((r_cavity / r), 4.0)); 
      Real rho = rho0 * cavity_fac; // flat nu -> flat rho outside cavity
      if (alpha != 0.0) rho *= Kokkos::pow(r, -1.5); // inhomo nu, update powerlaw

      // set pressure
      const Real cs_sqr = h_sqr * Gm_sum / (r + 1e-12);
      const Real P = cs_sqr * rho;

      // set velocity
      const Real v_phi = Kokkos::sqrt(Gm_sum / (r + 1e-12));
      const Real phi = Kokkos::atan2(x2v, x1v); // RH argument from +x axis
      const Real vx = -v_phi * Kokkos::sin(phi);
      const Real vy = v_phi * Kokkos::cos(phi);
      const Real vz = 0.0;
      
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

} 

// ======================== User-Defined Source Terms =========================
// ============================================================================

// write NBody data to hst output TODO: internalise as standard output
void NBodyHistory(HistoryData *pdata, Mesh *pm) {

  // generate labels for nbody data
  int num_nbody = pm->pmb_pack->pnbody->num_nbody;
  pdata->nhist = num_nbody * NVAR_HIST; 
  for (int n = 0; n < num_nbody; ++n) {
    int hist_offset = n * NVAR_HIST;
    pdata->label[0 + hist_offset] = "m" + std::to_string(n); 
    pdata->label[1 + hist_offset] = "x" + std::to_string(n);
    pdata->label[2 + hist_offset] = "y" + std::to_string(n);
    pdata->label[3 + hist_offset] = "z" + std::to_string(n);
    pdata->label[4 + hist_offset] = "vx" + std::to_string(n);
    pdata->label[5 + hist_offset] = "vy" + std::to_string(n);
    pdata->label[6 + hist_offset] = "vz" + std::to_string(n);
  } // end body loop

  // stash values
  for (int n = 0; n < num_nbody; ++n) {
    int hist_offset = n * NVAR_HIST;
    for (int i = 0; i < NVAR_HIST; i++) {
      pdata->hdata[i + hist_offset] = pm->pmb_pack->pnbody->nbody_data.h_view(n, i);
    } // end var loop
  } // end body loop
  return;
}