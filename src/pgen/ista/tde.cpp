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

Real pi = 3.141592653589793;
enum HistIndcies {M_BH = 0, X_BH = 1, Y_BH = 2, Z_BH = 3, VX_BH = 4, VY_BH = 5, VZ_BH = 6};
void FlybyHistory(HistoryData *pdata, Mesh *pm);

// ============================ Kokkos functions ==============================
// ============================================================================

KOKKOS_INLINE_FUNCTION
Real WrapTime(Real t, Real period) {
  return t - Kokkos::floor(t / period) * period;
}

KOKKOS_INLINE_FUNCTION
Real kokkos_atan2(Real y, Real x) {
#if defined(__CUDA_ARCH__)
  return atan2f(y, x);  
#else
  return std::atan2(y, x);
#endif
}

class Flyby {
  public:
    Flyby(Real q, Real e) {
      _mass_ratio = q;
      _eccentricity = e;

      // statics, for unitless system
      _semi_major_axis = 1.0;
      _primary_M = 1.0;
      _G_const = 1.0;

      // numerical method settinsgs
      _num_iter = 100;
      _precision = 1e-10;

      // static binary properties
      _mean_motion = Kokkos::sqrt(_G_const * _primary_M * (1 + _mass_ratio) / (Kokkos::pow(_semi_major_axis, 3)));
      _period = 2 * pi / _mean_motion;
    }
    
    // public access to private variables
    KOKKOS_INLINE_FUNCTION Real q() const { return _mass_ratio;}
    KOKKOS_INLINE_FUNCTION Real m_bin() const { return _primary_M * (1.0 + _mass_ratio);}
    KOKKOS_INLINE_FUNCTION Real m(int i) const {
      if (i == 0) {
        return _primary_M;
      } else {
        return _primary_M * _mass_ratio;
      }
    }

    // public methods
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<Kokkos::Array<Real, 3>, 2> BinaryPosition(Real t) {

      Real t_since_peri  = WrapTime(t, _period);
      Real a             = _semi_major_axis;
      Real e             = _eccentricity;
      Real q             = _mass_ratio;
      Real E             = solve_halleys(e, _mean_motion, t_since_peri);

      Real x_fac = a * (Kokkos::cos(E) - e) / (1.0 + q);
      Real y_fac = a * Kokkos::sqrt(1.0 - e * e) * Kokkos::sin(E) / (1.0 + q);
      Kokkos::Array<Real, 3> PrimaryPos   = { q * x_fac,  q * y_fac, 0.0};
      Kokkos::Array<Real, 3> SecondaryPos = {- x_fac, - y_fac, 0.0};
      return {PrimaryPos, SecondaryPos};
    }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<Real, 3> KeplerDerivatives(
      Real E, 
      Real e, 
      Real n, 
      Real t)
    {
      Real f = E - e * Kokkos::sin(E) - n * t;
      Real df = 1 - e * Kokkos::cos(E);
      Real ddf = e * Kokkos::sin(E);
      return {f, df, ddf};
    }

    KOKKOS_INLINE_FUNCTION
    Real solve_halleys(Real e, Real n, Real t) {
      
      int iter = 0;
      Real E = n * t;
      Kokkos::Array<Real, 3> f_df_ddf = KeplerDerivatives(E, e, n, t);
      
      while (Kokkos::abs(f_df_ddf[0]) > _precision){
          E -= f_df_ddf[0] * f_df_ddf[1] / (f_df_ddf[1] * f_df_ddf[1] - 0.5 * f_df_ddf[0] * f_df_ddf[2]);
          f_df_ddf = KeplerDerivatives(E, e, n, t);
          iter += 1;
          if (iter > _num_iter){
              return E;  
          }
        }
      return E;
    }

  private:
    // specified properties
    Real _mass_ratio, _eccentricity;

    // unitary properties
    Real _semi_major_axis, _primary_M, _G_const;
  
    // numerical method properties
    int _num_iter;
    Real _precision;

    // derived properties
    Real _mean_motion, _period;
};

namespace {
  std::unique_ptr<Flyby> h_flyby; // smart ptr to Flyby class on host
}

// TDE problem generator

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  
  // Binary parameters
  Real q_binary     = pin->GetOrAddReal("problem", "q_binary", 1.0);
  Real e_binary     = pin->GetOrAddReal("problem", "e_binary", 0.99);

  // Create binary class
  Flyby flyby     = Flyby(q_binary, e_binary);
  h_flyby         = std::make_unique<Flyby>(flyby);

  // Define source terms
  //g_sources_enabled = true;
  //user_srcs         = true;
  //user_srcs_func    = &TDESourceTerm;

  // Define history output
  user_hist         = true;
  user_hist_func    = &FlybyHistory;

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

  // (2) access prims from mesh block pack
  if (pmbp->phydro != nullptr) 
  {

    auto &w0_ = pmbp->phydro->w0;  // Primitive variables (density, velocity, pressure)

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

      // ===== Set primitive variables =====
      w0_(m, IDN, k, j, i) = 1.0;              // Density
      w0_(m, IVX, k, j, i) = 0.0;               // Velocity x-component
      w0_(m, IVY, k, j, i) = 0.0;               // Velocity y-component
      //w0_(m, IVZ, k, j, i) = 0.0;               // Velocity z-component
      w0_(m, IPR, k, j, i) = 1.0;             // Pressure
    }); 

    // ===== Convert primitives to conserved variables =====
    pmbp->phydro->peos->PrimToCons(w0_, pmbp->phydro->u0, is, ie, js, je, ks, ke);
  }  

} 

// ======================== User-Defined Source Terms =========================
// ============================================================================


void FlybyHistory(HistoryData *pdata, Mesh *pm) {
  // stores bh data into hist 
  
  // define storage labels
  int nvar_per_bh = 4;
  pdata->nhist = 2 * nvar_per_bh;
  for (int s = 0; s < 2; ++s) {
    int offset = s * nvar_per_bh;
    pdata->label[0 + offset] = "m" + std::to_string(s);
    pdata->label[1 + offset] = "x" + std::to_string(s);
    pdata->label[2 + offset] = "y" + std::to_string(s);
    pdata->label[3 + offset] = "z" + std::to_string(s);
  }

  // if data out of scope, set all to zero
  if (h_flyby == nullptr) {
    for (int n = 0; n < pdata->nhist; ++n) { pdata->hdata[n] = 0.0; }
    return;
  }

  // unpackage position data
  const auto pos_data = h_flyby->BinaryPosition(pm->time);

  // iterate over both minor and major bodies
  for (int s = 0; s < 2; ++s) {
    int offset = s * nvar_per_bh;
    pdata->hdata[M_BH + offset] = h_flyby->m(s);
    pdata->hdata[X_BH + offset] = pos_data[s][0];
    pdata->hdata[Y_BH + offset] = pos_data[s][1];
    pdata->hdata[Z_BH + offset] = pos_data[s][2];
  }
}