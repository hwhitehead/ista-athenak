// One or two constant-3D-entropy tori in Newtonian Cartesian hydro.
// Compile with -DPROBLEM=fluids/torus_init.
// In 2D, density and pressure mean vertically integrated Sigma and P.
// In 3D, they mean local rho and p; each torus can have its own tilt.
// The point potential -GM/r is used ONLY to construct the initial profiles.
// No gravity/source callback is enrolled: these tori expand when evolved.

#include <cmath>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"

namespace {
struct Torus {
  Real gm, rin, rp, s, n, beta, lp, hp, density_p;
  Real center[3], axis[3];
  bool three_d;

  // Integral of x^(a-1), including its logarithmic limit.
  KOKKOS_INLINE_FUNCTION
  Real Integral(Real x, Real lower, Real a) const {
    return Kokkos::abs(a) < 1.e-12 ? Kokkos::log(x/lower)
        : Kokkos::pow(lower, a)*Kokkos::expm1(a*Kokkos::log(x/lower))/a;
  }

  // Midplane enthalpy h(R), with h(rin)=0 and ell(R)=lp*(R/rp)^s.
  // 3D: h' = ell^2/R^3 - GM/R^2.
  // 2D: h' + beta*h/R = ell^2/R^3 - GM/R^2.
  KOKKOS_INLINE_FUNCTION
  Real Enthalpy(Real R) const {
    Real x = R/rp, a = rin/rp;
    return gm/rp*Kokkos::pow(x, -beta)
        *(Integral(x, a, 2*s+beta-2) - Integral(x, a, beta-1));
  }

  KOKKOS_INLINE_FUNCTION
  void Evaluate(Real x, Real y, Real z, Real &d, Real &p, Real v[3]) const {
    Real q[3] = {x-center[0], y-center[1], z-center[2]};
    Real r2 = q[0]*q[0] + q[1]*q[1] + q[2]*q[2];
    Real Z = axis[0]*q[0] + axis[1]*q[1] + axis[2]*q[2];
    Real R = Kokkos::sqrt(Kokkos::fmax(0.0, r2-Z*Z));
    d = p = 0;
    v[0] = v[1] = v[2] = 0;
    if (R <= rin) return;
    Real h = Enthalpy(R);
    // Exact vertical balance in the spherical point potential in 3D.
    if (three_d) h += gm/Kokkos::sqrt(r2) - gm/R;
    if (h <= 0) return;  // This also supplies the free outer edge.
    Real exponent = three_d ? n : n+0.5;
    d = density_p*Kokkos::pow(h/hp, exponent);
    if (!three_d) d *= Kokkos::pow(R/rp, 1.5);
    p = d*h/(exponent+1);
    Real omega = lp*Kokkos::pow(R/rp, s)/(R*R);
    // Azimuthal velocity = omega*(disk normal cross position).
    v[0] = omega*(axis[1]*q[2] - axis[2]*q[1]);
    v[1] = omega*(axis[2]*q[0] - axis[0]*q[2]);
    v[2] = omega*(axis[0]*q[1] - axis[1]*q[0]);
  }
};

struct TorusSet {
  Torus t[2];
  int count;
};
}  // namespace

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;  // Preserve the loaded conserved state.
  auto pack = pmy_mesh_->pmb_pack;
  if (!pack->phydro || pack->pmhd ||
      (!pmy_mesh_->two_d && !pmy_mesh_->three_d) ||
      pack->pcoord->is_special_relativistic || pack->pcoord->is_general_relativistic ||
      pack->pcoord->is_dynamical_relativistic)
    Kokkos::abort("torus_init requires Newtonian Cartesian hydro in 2D or 3D");
  auto eos = pack->phydro->peos->eos_data;
  if (!eos.is_ideal || eos.gamma <= 1)
    Kokkos::abort("torus_init requires an ideal gas with gamma > 1");
  bool three_d = pmy_mesh_->three_d;
  TorusSet disks{};
  disks.count = pin->GetOrAddInteger("problem", "num_tori", 1);
  if (disks.count < 1 || disks.count > 2) Kokkos::abort("num_tori must be 1 or 2");

  // Assume positive GM/density_p, 0 < rin < rp, and 0 <= s < 1/2.
  // For a finite outer edge also require:
  // rin/rp > [(1-beta)/(2-2*s-beta)]^(1/(1-2*s)).
  // rp fixes the maximum of integrated P in 2D, or midplane p in 3D.
  for (int a=0; a<disks.count; ++a) {
    auto &t = disks.t[a];
    std::string b = "problem_torus" + std::to_string(a+1);
    t.three_d = three_d;
    t.gm = pin->GetOrAddReal(b, "GM", 1);
    t.rin = pin->GetOrAddReal(b, "r_in", 20+6*a);
    t.rp = pin->GetOrAddReal(b, "r_p", 22+6*a);
    t.s = pin->GetOrAddReal(b, "ell_exponent", 0.2);
    t.density_p = pin->GetOrAddReal(b, "density_p", 1);
    t.center[0] = pin->GetOrAddReal(b, "x_center", 0);
    t.center[1] = pin->GetOrAddReal(b, "y_center", 0);
    t.center[2] = pin->GetOrAddReal(b, "z_center", 0);
    Real tilt = pin->GetOrAddReal(b, "tilt_deg", 0)*std::acos(-1.0)/180;
    Real node = pin->GetOrAddReal(b, "node_deg", 0)*std::acos(-1.0)/180;
    if (!three_d && (tilt != 0 || t.center[2] != 0))
      Kokkos::abort("2D tori must have tilt_deg=0 and z_center=0");
    t.axis[0] = std::sin(tilt)*std::cos(node);
    t.axis[1] = std::sin(tilt)*std::sin(node);
    t.axis[2] = std::cos(tilt);
    t.n = 1/(eos.gamma-1);
    t.beta = three_d ? 0 : 1.5/(t.n+1.5);
    t.lp = std::sqrt(t.gm*t.rp);
    t.hp = t.Enthalpy(t.rp);
  }

  auto ind = pmy_mesh_->mb_indcs;
  auto size = pack->pmb->mb_size;
  auto u = pack->phydro->u0;
  int nvars = pack->phydro->nhydro + pack->phydro->nscalars;
  Real gm1 = eos.gamma-1, dfloor = eos.dfloor, pfloor = eos.pfloor;
  par_for("initialize_tori", DevExeSpace(), 0, pack->nmb_thispack-1,
          ind.ks, ind.ke, ind.js, ind.je, ind.is, ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-ind.is, ind.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    Real y = CellCenterX(j-ind.js, ind.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real z = three_d ? CellCenterX(k-ind.ks, ind.nx3,
                                  size.d_view(m).x3min, size.d_view(m).x3max) : 0;
    Real d = dfloor, p = pfloor, v[3] = {0, 0, 0};
    // Start with static floor gas; the denser torus supplies overlapping cells.
    for (int a=0; a<disks.count; ++a) {
      Real td, tp, tv[3];
      disks.t[a].Evaluate(x, y, z, td, tp, tv);
      if (td > d) {
        d = td;
        p = Kokkos::fmax(tp, pfloor);
        for (int b=0; b<3; ++b) v[b] = tv[b];
      }
    }
    for (int n=0; n<nvars; ++n) u(m, n, k, j, i) = 0;
    u(m, IDN, k, j, i) = d;
    u(m, IM1, k, j, i) = d*v[0];
    u(m, IM2, k, j, i) = d*v[1];
    u(m, IM3, k, j, i) = d*v[2];
    u(m, IEN, k, j, i) = p/gm1 + 0.5*d*(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  });
}
