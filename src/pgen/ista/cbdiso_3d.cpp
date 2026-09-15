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


// ============================ Kokkos functions ==============================
// ============================================================================

KOKKOS_INLINE_FUNCTION
Real wrap_time(Real t, Real period) {
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



// ================================ Binary ====================================
// ============================================================================

class Binary {
  public:
    Binary(Real semimajoraxis, 
           Real eccentricity, 
           Real mass_ratio,
           Real TotalMass, 
           Real SinkRadius, 
           Real SinkRate,
           Real SofteningRadius,
           int SinkModel)
      : _semimajoraxis(semimajoraxis),
        _eccentricity(eccentricity),
        _mass_ratio(mass_ratio),
        _TotalMass(TotalMass),
        _SinkRadius(SinkRadius),
        _SinkRate(SinkRate),
        _SofteningRadius(SofteningRadius),
        _SinkType(SinkModel),
        _period(Period()),
        _mean_motion(MeanMotion()) {}

    // public properties 
    KOKKOS_INLINE_FUNCTION Real TotalMass() const { return _TotalMass; }
    KOKKOS_INLINE_FUNCTION Real m1()        const { return _TotalMass               / (1 + _mass_ratio); }
    KOKKOS_INLINE_FUNCTION Real m2()        const { return _TotalMass * _mass_ratio / (1 + _mass_ratio); }
    KOKKOS_INLINE_FUNCTION Real rsink1()    const { return _SinkRadius; }
    KOKKOS_INLINE_FUNCTION Real rsink2()    const { return _SinkRadius*_mass_ratio; }
    KOKKOS_INLINE_FUNCTION Real rsoft1()    const { return _SofteningRadius; }
    KOKKOS_INLINE_FUNCTION Real rsoft2()    const { return _SofteningRadius * _mass_ratio; }
    KOKKOS_INLINE_FUNCTION int SinkModel()  const { return _SinkType; }
    KOKKOS_INLINE_FUNCTION Real SinkRate()  const { return _SinkRate; }


    KOKKOS_INLINE_FUNCTION
    Real OmegaTilde(
        Real x, 
        Real y, 
        Real z, 
        Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position) const 
      {
        Real G                        = 1.0;
        Kokkos::Array<Real, 3> r1_vec = {x -  binary_position[0][0], y -  binary_position[0][1], z -  binary_position[0][2]};
        Kokkos::Array<Real, 3> r2_vec = {x -  binary_position[1][0], y -  binary_position[1][1], z -  binary_position[1][2]};
        Real r1                       = Kokkos::sqrt(r1_vec[0]*r1_vec[0] + r1_vec[1]*r1_vec[1] + r1_vec[2]*r1_vec[2]);
        Real r2                       = Kokkos::sqrt(r2_vec[0]*r2_vec[0] + r2_vec[1]*r2_vec[1] + r2_vec[2]*r2_vec[2]);
        Real omega1_sq                = G * m1() / (r1*r1*r1 + rsoft1()*rsoft1());
        Real omega2_sq                = G * m2() / (r2*r2*r2 + rsoft2()*rsoft2());
        return Kokkos::sqrt(omega1_sq + omega2_sq);
      }


    KOKKOS_INLINE_FUNCTION
    Real GravitationalPotential(
      Real x, 
      Real y, 
      Real z, 
      Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position) const 
    {
      Real G                        = 1.0;
      Kokkos::Array<Real, 3> r1_vec = {x - binary_position[0][0], y - binary_position[0][1], z - binary_position[0][2]};
      Kokkos::Array<Real, 3> r2_vec = {x - binary_position[1][0], y - binary_position[1][1], z - binary_position[1][2]};
      Real phi1                     = -G * m1() / Kokkos::sqrt(r1_vec[0]*r1_vec[0] + r1_vec[1]*r1_vec[1] + r1_vec[2]*r1_vec[2] + rsoft1()*rsoft1());
      Real phi2                     = -G * m2() / Kokkos::sqrt(r2_vec[0]*r2_vec[0] + r2_vec[1]*r2_vec[1] + r2_vec[2]*r2_vec[2] + rsoft2()*rsoft2());
      return phi1 + phi2;
    }


    KOKKOS_INLINE_FUNCTION
    Real KeplerEquation(
      Real E, 
      Real e, 
      Real n, 
      Real t)
    {
      return E - e * Kokkos::sin(E) - n * t;
    }


    KOKKOS_INLINE_FUNCTION
    Real solve_newton_rapheson(
      Real e, 
      Real n, 
      Real t)
    {
      int iter   = 0;
      Real E     = n * t;
      Real func  = KeplerEquation(E, e, n, t); 
      Real deriv = 1 - e * Kokkos::cos(E);
      
      while (Kokkos::abs(func) > 1e-15){
          E    -= func / deriv;
          iter += 1;
          if (iter > 10){
              return E;  
          }
        }
      return E;
    }

    // Right now coplanar fix in general soon
    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<Kokkos::Array<Real, 3>, 2> BinaryPosition(
      Real t)
    {
      Real time_periapse = wrap_time(t, _period);
      Real a             = _semimajoraxis;
      Real e             = _eccentricity;
      Real q             = _mass_ratio;
      Real E             = solve_newton_rapheson(e, _mean_motion, time_periapse);

      Kokkos::Array<Real, 3> PrimaryPos   = { a*q/(1+q)*(Kokkos::cos(E)-e),  a*q/(1+q)*Kokkos::sqrt(1-e*e)*Kokkos::sin(E), 0.0};
      Kokkos::Array<Real, 3> SecondaryPos = {-a  /(1+q)*(Kokkos::cos(E)-e), -a  /(1+q)*Kokkos::sqrt(1-e*e)*Kokkos::sin(E), 0.0};
      return {PrimaryPos, SecondaryPos};
    }


    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<Kokkos::Array<Real, 3>, 2> BinaryVelocity(
      Real t)
    {
      Real a             = _semimajoraxis;
      Real e             = _eccentricity;
      Real q             = _mass_ratio;
      Real time_periapse = wrap_time(t, _period);
      Real E             = solve_newton_rapheson(e, _mean_motion, time_periapse);
      Real Edot          = _mean_motion / (1 - e * Kokkos::cos(E));
      
      Kokkos::Array<Real, 3> PrimaryVel   = {-a*q/(1+q)*Edot*Kokkos::sin(E),  a*q/(1+q)*Kokkos::sqrt(1-e*e)*Edot*Kokkos::cos(E), 0.0};
      Kokkos::Array<Real, 3> SecondaryVel = { a  /(1+q)*Edot*Kokkos::sin(E), -a  /(1+q)*Kokkos::sqrt(1-e*e)*Edot*Kokkos::cos(E), 0.0};
      return {PrimaryVel, SecondaryVel};
    }
    
    // live binary implementation?

  private:
    Real _semimajoraxis;
    Real _eccentricity;
    Real _mass_ratio;
    Real _TotalMass;
    Real _SinkRadius;
    Real _SinkRate;
    Real _SofteningRadius;
    Real _mean_motion;
    Real _period;
    int _SinkType; 

    KOKKOS_INLINE_FUNCTION Real MeanMotion() const { return Kokkos::sqrt(TotalMass() / (_semimajoraxis*_semimajoraxis*_semimajoraxis)); }
    KOKKOS_INLINE_FUNCTION Real Period()     const { return 2 * 3.141592653589793 * Kokkos::sqrt(Kokkos::pow(_semimajoraxis, 3) / TotalMass()); }
}; 



// ================================== Disk ====================================
// ============================================================================

class Disk {
  public:
    Disk(Real Mach, 
        Real alpha, 
        const Binary& binary)
      : _Mach(Mach),
        _alpha(alpha),
        _binary(binary) {}
    
    

    KOKKOS_INLINE_FUNCTION
    Real SoundSpeedSquare(
      Real x, 
      Real y, 
      Real z,
      Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position) const 
    {
      Real cs2 = - _binary.GravitationalPotential(x, y, z, binary_position) / (_Mach * _Mach);
      return cs2;
    }


    KOKKOS_INLINE_FUNCTION
    Real Viscosity(
      Real x, 
      Real y, 
      Real z, 
      Real t,
      Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position) const 
    {
      Real cs    = Kokkos::sqrt(SoundSpeedSquare(x, y, z, binary_position));
      Real H     = cs / _binary.OmegaTilde(x, y, z, binary_position);  // Scale height H = cs / Omega
      return _alpha * cs * H;
    }


    KOKKOS_INLINE_FUNCTION
    Real InitialDensity_2D(
      Real r, 
      Real z,
      Real rcav) const 
    {
      Real G        = 1.0;
      Real GM       = G * _binary.TotalMass();      
      Real rho0     = 1.0;                           // Density normalization
      Real h2       = 1.0 / _Mach / _Mach;           // Aspect ratio 
      Real cs2      = h2 * GM / (r+1e-6);            // arbitrary softening only for IC
      Real vertical = Kokkos::exp((1/h2) * (1/Kokkos::sqrt(1.0 + z*z/(r*r+1e-6)) - 1.0));  
      Real cavity   = 0.0001 + 0.9999 * Kokkos::exp(-Kokkos::pow((rcav / r), 4.0)); 
      return rho0 * Kokkos::pow(r, -1.5) * vertical * cavity;
    }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<Real, 3> InitialVelocity_2D(
      Real x, 
      Real y, 
      Real z) const 
    {
      Real G     = 1.0;
      Real r_cyl = Kokkos::sqrt(x*x + y*y);
      Real phi   = kokkos_atan2(y, x);
      Real v_phi = Kokkos::sqrt(G * _binary.TotalMass() / (r_cyl + 1e-6));  // Keplerian velocity
      Real vx    = -v_phi * Kokkos::sin(phi);
      Real vy    =  v_phi * Kokkos::cos(phi);
      Real vz    = 0.0;   // No vertical motion
      return {vx, vy, vz};
    }

    KOKKOS_INLINE_FUNCTION
    Real InitialDensity(
      Real x, 
      Real y, 
      Real z, 
      Real Orientation,
      Real rcav) const 
    {
      // Rotate (x, z) by Orientation about y-axis
      Real x_rot = x * Kokkos::cos(Orientation) - z * Kokkos::sin(Orientation);
      Real z_rot = x * Kokkos::sin(Orientation) + z * Kokkos::cos(Orientation);
      Real r_cyl = Kokkos::sqrt(y*y + x_rot*x_rot);
      Real z_cyl = z_rot;
            
      // // Rotate (y, z) by Orientation about x-axis
      // Real y_rot = y * Kokkos::cos(Orientation) - z * Kokkos::sin(Orientation);
      // Real z_rot = y * Kokkos::sin(Orientation) + z * Kokkos::cos(Orientation);
      // Real r_cyl = Kokkos::sqrt(x*x + y_rot*y_rot);
      // Real z_cyl = z_rot;
      return InitialDensity_2D(r_cyl, z_cyl, rcav);
    }

    KOKKOS_INLINE_FUNCTION
    Kokkos::Array<Real, 3> InitialVelocity(
      Real x, 
      Real y, 
      Real z, 
      Real Orientation) const 
    {
      Real x_rot = x * Kokkos::cos(Orientation) - z * Kokkos::sin(Orientation);
      Real z_rot = x * Kokkos::sin(Orientation) + z * Kokkos::cos(Orientation);
      //Real r_cyl = Kokkos::sqrt(y*y + x_rot*x_rot);
      //Real z_cyl = z_rot;

      Kokkos::Array<Real, 3> vel_disk = InitialVelocity_2D(x_rot, y, z_rot);

      Real vx = vel_disk[0] * Kokkos::cos(Orientation); // vel_disk[2] is always 0
      Real vy = vel_disk[1]; 
      Real vz = -vel_disk[0] * Kokkos::sin(Orientation);
      
      // Real y_rot = y * Kokkos::cos(Orientation) - z * Kokkos::sin(Orientation);
      // Real z_rot = y * Kokkos::sin(Orientation) + z * Kokkos::cos(Orientation);
      // Kokkos::Array<Real, 3> vel_disk = InitialVelocity_2D(x, y_rot, z_rot); 

      // Real vx = vel_disk[0];
      // Real vy = vel_disk[1] * Kokkos::cos(Orientation); // vel_disk[2] is always 0
      // Real vz = vel_disk[1] * Kokkos::sin(Orientation);
      return {vx, vy, vz};
    }


  private:
    Real _Mach;
    Real _alpha;
    Binary _binary;
  };



// =========================== Forward Declarations ===========================
// ============================================================================

void viscous_source_term(Mesh *pm, const Real beta_dt, Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position);
void binary_source_term(Mesh *pm, const Real beta_dt,  Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position, Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_velocity);
void enforce_isothermal_pressure(Mesh *pm);
void enforce_floors(Mesh *pm);
void CircumbinaryHistory(HistoryData *pdata, Mesh *pm);
void CircumbinaryFinalAnalysis(ParameterInput *pin, Mesh *pm);
void DiagnosticPrint(Mesh *pm);  // TEMPORARY DEBUG — remove once blowup is diagnosed


namespace {
  Real g_mach;
  Real g_alpha;
  Real g_floor_density;
  Real g_floor_pressure;
  bool g_sources_enabled;
  std::unique_ptr<Binary>  g_binary;
  Kokkos::View<Real[2]>    g_mass_accreted;      // cumulative mass gained by each BH
  Kokkos::View<Real[2][3]> g_momentum_grav;      // cumulative momentum gained (gravity)
  Kokkos::View<Real[2][3]> g_momentum_accreted;  // cumulative momentum gained (accretion)
  Kokkos::View<Real[2]>    g_energy_grav;        // cumulative energy gained (gravity)
  Kokkos::View<Real[2]>    g_energy_accreted;    // cumulative energy gained (accretion)
  Real g_last_reset_time;  // sim time at which the accumulators above were last zeroed
}
//  Kokkos::View<Real[2]> is a 1D array of Real with size 2 designed to be easy
//  to hand between host and device. Access uses function call syntax ()



// =============================== Source Terms ===============================
// ============================================================================

void circumbinary_source_term(
  Mesh *pm, 
  const Real beta_dt_local) 
{
  Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position = g_binary->BinaryPosition(pm->time);
  Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_velocity = g_binary->BinaryVelocity(pm->time);
  viscous_source_term(pm, beta_dt_local, binary_position);
  binary_source_term(pm , beta_dt_local, binary_position, binary_velocity);
  enforce_floors(pm);
  enforce_isothermal_pressure(pm);
}



// =================== Circumbinary Disk Problem Generator ====================
// ============================================================================

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  
  // Binary parameters
  Real a_binary     = pin->GetOrAddReal("problem", "a_binary"     , 1.0);
  Real q_mass       = pin->GetOrAddReal("problem", "q_mass"       , 1.0);
  Real eccentricity = pin->GetOrAddReal("problem", "eccentricity" , 0.0);
  Real r_soft       = pin->GetOrAddReal("problem", "r_soft"       , 0.05);
  Real r_sink       = pin->GetOrAddReal("problem", "r_sink"       , 0.05);
  Real m_total      = pin->GetOrAddReal("problem", "m_total"      , 1.0);
  Real sink_rate    = pin->GetOrAddReal("problem", "sink_rate"    , 1.0);
  int SinkModel     = pin->GetOrAddInteger("problem", "sink_model", 2  );

  // Disk parameters
  g_mach            = pin->GetOrAddReal("problem", "Mach"          , 10.0);
  g_alpha           = pin->GetOrAddReal("problem", "alpha"         , 0.1 );
  Real Orientation  = pin->GetOrAddReal("problem", "orientation"   , 0.0 ) * 0.017453292519943295; // in degrees
  Real rcav         = pin->GetOrAddReal("problem", "cavity_radius" , 1.0 );

  // numerical parameters
  g_floor_density   = pin->GetOrAddReal("problem", "floor_density"  , 1e-10);
  g_floor_pressure  = pin->GetOrAddReal("problem", "floor_pressure" , 1e-12);


  // Create binary class
  Binary binary     = Binary(a_binary, eccentricity, q_mass, m_total, r_sink, sink_rate, r_soft, SinkModel);
  g_binary          = std::make_unique<Binary>(binary);

  // Create disk class
  Disk disk         = Disk(g_mach, g_alpha, *g_binary);
  //g_disk            = std::make_unique<Disk>(disk);
  const auto binary_position0 = g_binary->BinaryPosition(0.0);


  // Allocate and zero binary accretion/torque/power diagnostics accumulators
  g_mass_accreted     = Kokkos::View<Real[2]>("mass_accreted");
  g_momentum_grav     = Kokkos::View<Real[2][3]>("p_grav");
  g_momentum_accreted = Kokkos::View<Real[2][3]>("p_accr");
  g_energy_grav       = Kokkos::View<Real[2]>("e_grav");
  g_energy_accreted   = Kokkos::View<Real[2]>("e_accr");
  Kokkos::deep_copy(g_mass_accreted, 0.0); // fills every slot with 0.0
  Kokkos::deep_copy(g_momentum_grav, 0.0);
  Kokkos::deep_copy(g_momentum_accreted, 0.0);
  Kokkos::deep_copy(g_energy_grav, 0.0);
  Kokkos::deep_copy(g_energy_accreted, 0.0);

  g_last_reset_time = pmy_mesh_->time;

  // Define source terms
  g_sources_enabled = true;
  user_srcs         = true;
  user_srcs_func    = &circumbinary_source_term;

  // Define history output
  user_hist         = true;
  user_hist_func    = &CircumbinaryHistory;

  // Free the Kokkos::View accumulators before Kokkos::finalize() runs (they have
  // static storage duration, so they'd be destroyed after main() returns).
  pgen_final_func   = &CircumbinaryFinalAnalysis;

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
    par_for("pgen_circumbinary", 
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


      // ===== Compute density =====
      Real rho      = disk.InitialDensity(x1v, x2v, x3v, Orientation, rcav);

      // ===== Compute velocity =====
      Kokkos::Array<Real, 3> vel  = disk.InitialVelocity(x1v, x2v, x3v, Orientation);
      Real vx       = vel[0];
      Real vy       = vel[1];
      Real vz       = vel[2];
      
      // ===== Compute pressure =====
      Real cs2      = disk.SoundSpeedSquare(x1v, x2v, x3v, binary_position0);
      Real pres     = rho * cs2;

      // ===== Set primitive variables =====
      w0_(m, IDN, k, j, i) = rho;              // Density
      w0_(m, IVX, k, j, i) = vx;               // Velocity x-component
      w0_(m, IVY, k, j, i) = vy;               // Velocity y-component
      w0_(m, IVZ, k, j, i) = vz;               // Velocity z-component
      w0_(m, IPR, k, j, i) = pres;             // Pressure

    }); 

    // ===== Convert primitives to conserved variables =====
    pmbp->phydro->peos->PrimToCons(w0_, pmbp->phydro->u0, is, ie, js, je, ks, ke);

  }  

} 



// ======================== User-Defined Source Terms =========================
// ============================================================================


void viscous_source_term(
  Mesh *pm, 
  const Real beta_dt, 
  Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position) 
{
  
  if (!g_sources_enabled || beta_dt <= 0.0) {
    return;
  }

  if (pm == nullptr || pm->pmb_pack == nullptr || g_binary == nullptr) {
    return;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) {
    return;
  }

  const Disk disk(g_mach, g_alpha, *g_binary);

  auto &indcs              = pm->mb_indcs;
  auto &size               = pmbp->pmb->mb_size;
  auto &w                  = pmbp->phydro->w0;
  auto &u                  = pmbp->phydro->u0;
  const bool has_y         = (indcs.nx2 > 1);
  const bool has_z         = (indcs.nx3 > 1);
  const bool update_energy = pmbp->phydro->peos->eos_data.is_ideal;
  const Real t_now         = pm->time;
  const Real beta_dt_local = beta_dt;

  par_for("disk_viscous_source",
          DevExeSpace(),
          0, (pmbp->nmb_thispack - 1),
          indcs.ks, indcs.ke,
          indcs.js, indcs.je,
          indcs.is, indcs.ie,
          KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real dx1 = size.d_view(m).dx1;
    const Real dx2 = size.d_view(m).dx2;
    const Real dx3 = size.d_view(m).dx3;

    const int ip = (i == indcs.ie) ? i : (i + 1);
    const int im = (i == indcs.is) ? i : (i - 1);
    const int jp = (j == indcs.je) ? j : (j + 1);
    const int jm = (j == indcs.js) ? j : (j - 1);
    const int kp = (k == indcs.ke) ? k : (k + 1);
    const int km = (k == indcs.ks) ? k : (k - 1);

    const Real inv_dx1_sq = 1.0 / (dx1 * dx1);
    const Real inv_dx2_sq = has_y ? 1.0 / (dx2 * dx2) : 0.0;
    const Real inv_dx3_sq = has_z ? 1.0 / (dx3 * dx3) : 0.0;

    const Real inv_4dx1dx2 = has_y ? 1.0 / (4.0 * dx1 * dx2) : 0.0;
    const Real inv_4dx1dx3 = has_z ? 1.0 / (4.0 * dx1 * dx3) : 0.0;
    const Real inv_4dx2dx3 = (has_y && has_z) ? 1.0 / (4.0 * dx2 * dx3) : 0.0;

    const Real rho = w(m, IDN, k, j, i);

    const Real vx = w(m, IVX, k, j, i);
    const Real vy = w(m, IVY, k, j, i);
    const Real vz = w(m, IVZ, k, j, i);

    const Real second_x_vx = (w(m, IVX, k, j, ip) - 2.0 * vx + w(m, IVX, k, j, im)) * inv_dx1_sq;
    const Real second_y_vx = has_y ? (w(m, IVX, k, jp, i) - 2.0 * vx + w(m, IVX, k, jm, i)) * inv_dx2_sq : 0.0;
    const Real second_z_vx = has_z ? (w(m, IVX, kp, j, i) - 2.0 * vx + w(m, IVX, km, j, i)) * inv_dx3_sq : 0.0;

    const Real second_x_vy = (w(m, IVY, k, j, ip) - 2.0 * vy + w(m, IVY, k, j, im)) * inv_dx1_sq;
    const Real second_y_vy = has_y ? (w(m, IVY, k, jp, i) - 2.0 * vy + w(m, IVY, k, jm, i)) * inv_dx2_sq : 0.0;
    const Real second_z_vy = has_z ? (w(m, IVY, kp, j, i) - 2.0 * vy + w(m, IVY, km, j, i)) * inv_dx3_sq : 0.0;

    const Real second_x_vz = (w(m, IVZ, k, j, ip) - 2.0 * vz + w(m, IVZ, k, j, im)) * inv_dx1_sq;
    const Real second_y_vz = has_y ? (w(m, IVZ, k, jp, i) - 2.0 * vz + w(m, IVZ, k, jm, i)) * inv_dx2_sq : 0.0;
    const Real second_z_vz = has_z ? (w(m, IVZ, kp, j, i) - 2.0 * vz + w(m, IVZ, km, j, i)) * inv_dx3_sq : 0.0;

    // Mixed partial derivatives (central differences on the diagonal neighbors),
    // needed to build the full grad(div v) vector rather than just its diagonal part.
    const Real d2vx_dxdy = has_y ? (w(m, IVX, k, jp, ip) - w(m, IVX, k, jp, im)
                                   - w(m, IVX, k, jm, ip) + w(m, IVX, k, jm, im)) * inv_4dx1dx2 : 0.0;
    const Real d2vx_dxdz = has_z ? (w(m, IVX, kp, j, ip) - w(m, IVX, kp, j, im)
                                   - w(m, IVX, km, j, ip) + w(m, IVX, km, j, im)) * inv_4dx1dx3 : 0.0;

    const Real d2vy_dxdy = has_y ? (w(m, IVY, k, jp, ip) - w(m, IVY, k, jp, im)
                                   - w(m, IVY, k, jm, ip) + w(m, IVY, k, jm, im)) * inv_4dx1dx2 : 0.0;
    const Real d2vy_dydz = (has_y && has_z) ? (w(m, IVY, kp, jp, i) - w(m, IVY, kp, jm, i)
                                              - w(m, IVY, km, jp, i) + w(m, IVY, km, jm, i)) * inv_4dx2dx3 : 0.0;

    const Real d2vz_dxdz = has_z ? (w(m, IVZ, kp, j, ip) - w(m, IVZ, kp, j, im)
                                   - w(m, IVZ, km, j, ip) + w(m, IVZ, km, j, im)) * inv_4dx1dx3 : 0.0;
    const Real d2vz_dydz = (has_y && has_z) ? (w(m, IVZ, kp, jp, i) - w(m, IVZ, kp, jm, i)
                                              - w(m, IVZ, km, jp, i) + w(m, IVZ, km, jm, i)) * inv_4dx2dx3 : 0.0;

    const Real lap_vx = second_x_vx + second_y_vx + second_z_vx;
    const Real lap_vy = second_x_vy + second_y_vy + second_z_vy;
    const Real lap_vz = second_x_vz + second_y_vz + second_z_vz;

    // Full grad(div v) components: diagonal term plus the
    // cross-derivative terms that were previously missing.
    const Real grad_div_x = second_x_vx + d2vy_dxdy + d2vz_dxdz;
    const Real grad_div_y = d2vx_dxdy + second_y_vy + d2vz_dydz;
    const Real grad_div_z = d2vx_dxdz + d2vy_dydz + second_z_vz;

    const Real x1v = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real x2v = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real x3v = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);

    const Real nu    = disk.Viscosity(x1v, x2v, x3v, t_now, binary_position);
    const Real eta   = rho * nu;
    const Real coeff = 1.0 / 3.0;

    const Real ax = eta * (lap_vx + coeff * grad_div_x);
    const Real ay = eta * (lap_vy + coeff * grad_div_y);
    const Real az = eta * (lap_vz + coeff * grad_div_z);

    u(m, IVX, k, j, i) += beta_dt_local * ax;
    u(m, IVY, k, j, i) += beta_dt_local * ay;
    u(m, IVZ, k, j, i) += beta_dt_local * az;

    if (update_energy) {
      const Real visc_work = vx * ax + vy * ay + vz * az;
      u(m, IEN, k, j, i) += beta_dt_local * visc_work;
    }
  });
}


KOKKOS_INLINE_FUNCTION
void point_mass_source_term(
  const Kokkos::Array<Kokkos::Array<Real, 3>, 2> &binary_velocity,
  const Kokkos::Array<Kokkos::Array<Real, 3>, 2> &binary_position,
  const Real x,
  const Real y,
  const Real z,
  const Real dt,
  const Real rho,
  const Real vx,
  const Real vy,
  const Real vz,
  const int which_mass,
  const Binary &binary,
  const Kokkos::View<Real[2]> &mass_accreted,
  const Kokkos::View<Real[2][3]> &p_grav,
  const Kokkos::View<Real[2][3]> &p_accr,
  const Kokkos::View<Real[2]> &e_grav,
  const Kokkos::View<Real[2]> &e_accr,
  Real delta_cons[5])
{
    for (int n = 0; n < 5; ++n) {
      delta_cons[n] = 0.0;
    }

    if (dt <= 0.0 || rho <= 0.0) {
      return;
    }

    const bool primary          = (which_mass == 1);
    const Real sink_radius      = primary ? binary.rsink1() : binary.rsink2();
    const Real softening_radius = primary ? binary.rsoft1() : binary.rsoft2();
    const Real mass             = primary ? binary.m1()     : binary.m2();
    const auto &pos             = binary_position[which_mass - 1];
    const auto &vel             = binary_velocity[which_mass - 1];
    const int s                 = which_mass - 1;

    const Real x0 = pos[0];
    const Real y0 = pos[1];
    const Real z0 = pos[2];
    const Real dx = x - x0;
    const Real dy = y - y0;
    const Real dz = z - z0;
    const Real r2 = dx * dx + dy * dy + dz * dz;
    const Real dr = Kokkos::sqrt(r2);

    Real sink_rate = 0.0;
    if (sink_radius > 0.0 && dr < 2.0 * sink_radius) {
      const Real ratio = dr / sink_radius;
      sink_rate = binary.SinkRate() * Kokkos::exp(-Kokkos::pow(ratio, 4.0));
    }
    if (dt > 0.0 && sink_rate > 0.0) {
      sink_rate = Kokkos::min(sink_rate, 0.9 / dt);
    }

    const Real fgrav_numerator = rho * mass * Kokkos::pow(r2 + softening_radius * softening_radius, -1.5);
    const Real fx              = -fgrav_numerator * dx;
    const Real fy              = -fgrav_numerator * dy;
    const Real fz              = -fgrav_numerator * dz;
    const Real mdot            = -rho * sink_rate;

    if (binary.SinkModel() == 0) {
      return;
    }

    Real accr_px, accr_py, accr_pz;   // momentum given to the GAS by the sink term

    if (binary.SinkModel() == 1) {
      // acceleration free
      accr_px = dt * mdot * vx;
      accr_py = dt * mdot * vy;
      accr_pz = dt * mdot * vz;

    } else if (binary.SinkModel() == 2) {
      // torque free
      const Real vx0       = vel[0];
      const Real vy0       = vel[1];
      const Real vz0       = vel[2];
      const Real inv_dr    = 1.0 / (dr + 1e-12);
      const Real rhatx     = dx * inv_dr;
      const Real rhaty     = dy * inv_dr;
      const Real rhatz     = dz * inv_dr;
      const Real dvdotrhat = (vx - vx0) * rhatx + (vy - vy0) * rhaty + (vz - vz0) * rhatz;
      const Real vxstar    = dvdotrhat * rhatx + vx0;
      const Real vystar    = dvdotrhat * rhaty + vy0;
      const Real vzstar    = dvdotrhat * rhatz + vz0;

      accr_px = dt * mdot * vxstar;
      accr_py = dt * mdot * vystar;
      accr_pz = dt * mdot * vzstar;

    } else {
      return;
    }

    const Real grav_px = dt * fx;
    const Real grav_py = dt * fy;
    const Real grav_pz = dt * fz;

    delta_cons[0] = dt * mdot;
    delta_cons[1] = grav_px + accr_px;
    delta_cons[2] = grav_py + accr_py;
    delta_cons[3] = grav_pz + accr_pz;
    delta_cons[4] = 0.0;

    // record what was actually applied, in the BH frame
    // atomic add is a safe way to update when many are accessing
    Kokkos::atomic_add(&mass_accreted(s), -dt * mdot);
    Kokkos::atomic_add(&p_grav(s, 0), -grav_px);
    Kokkos::atomic_add(&p_grav(s, 1), -grav_py);
    Kokkos::atomic_add(&p_grav(s, 2), -grav_pz);
    Kokkos::atomic_add(&p_accr(s, 0), -accr_px);
    Kokkos::atomic_add(&p_accr(s, 1), -accr_py);
    Kokkos::atomic_add(&p_accr(s, 2), -accr_pz);

    // Power delivered to the BH = (force on BH) . (BH's own velocity), impulse form.
    // vel is the BH's current velocity (binary_velocity[which_mass-1], grabbed above).
    Kokkos::atomic_add(&e_grav(s), -(grav_px*vel[0] + grav_py*vel[1] + grav_pz*vel[2]));
    Kokkos::atomic_add(&e_accr(s), -(accr_px*vel[0] + accr_py*vel[1] + accr_pz*vel[2]));
}


void binary_source_term(
  Mesh *pm, 
  const Real beta_dt, 
  Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_position, 
  Kokkos::Array<Kokkos::Array<Real, 3>, 2> binary_velocity) 
{
  if (!g_sources_enabled || beta_dt <= 0.0) {
    return;
  }

  if (pm == nullptr || pm->pmb_pack == nullptr || g_binary == nullptr) {
    return;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) {
    return;
  }

  const Binary binary      = *g_binary;   // save globals to host for GPU scope
  auto &indcs              = pm->mb_indcs;
  auto &size               = pmbp->pmb->mb_size;
  auto &w                  = pmbp->phydro->w0;
  auto &u                  = pmbp->phydro->u0;
  const bool update_energy = pmbp->phydro->peos->eos_data.is_ideal;
  const Real beta_dt_local = beta_dt;
  auto mass_accreted       = g_mass_accreted;  
  auto p_grav              = g_momentum_grav;
  auto p_accr              = g_momentum_accreted;
  auto e_grav              = g_energy_grav;
  auto e_accr              = g_energy_accreted;

  par_for(
    "binary_sink_source",
    DevExeSpace(),
    0, (pmbp->nmb_thispack - 1),
    indcs.ks, indcs.ke,
    indcs.js, indcs.je,
    indcs.is, indcs.ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) 
    {
      const Real x1v = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
      const Real x2v = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
      const Real x3v = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);

      const Real rho = w(m, IDN, k, j, i);
      const Real vx  = w(m, IVX, k, j, i);
      const Real vy  = w(m, IVY, k, j, i);
      const Real vz  = w(m, IVZ, k, j, i);

      Real delta_total[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

      for (int which_mass = 1; which_mass <= 2; ++which_mass) {
        Real delta_cons[5];
        point_mass_source_term(
          binary_velocity,
          binary_position,
          x1v,
          x2v,
          x3v,
          beta_dt_local,
          rho,
          vx,
          vy,
          vz,
          which_mass,
          binary,
          mass_accreted,
          p_grav,
          p_accr,
          e_grav,
          e_accr,
          delta_cons);

        for (int n = 0; n < 5; ++n) {
          delta_total[n] += delta_cons[n];
        }
      }

      u(m, IDN, k, j, i) += delta_total[0];
      u(m, IVX, k, j, i) += delta_total[1];
      u(m, IVY, k, j, i) += delta_total[2];
      u(m, IVZ, k, j, i) += delta_total[3];
      if (update_energy) {
        u(m, IEN, k, j, i) += delta_total[4];
      }
  });
}



void enforce_isothermal_pressure(Mesh *pm) {
  if (!g_sources_enabled) {
    return;
  }

  if (pm == nullptr || pm->pmb_pack == nullptr || g_binary == nullptr) {
    return;
  }

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) {
    return;
  }

  
  const Disk disk(g_mach, g_alpha, *g_binary);   // save globals to host for GPU scope
  const auto binary_position = g_binary->BinaryPosition(pm->time);

  auto &indcs              = pm->mb_indcs;
  auto &size               = pmbp->pmb->mb_size;
  auto &w                  = pmbp->phydro->w0;
  auto &u                  = pmbp->phydro->u0;
  const bool update_energy = pmbp->phydro->peos->eos_data.is_ideal;
  const Real gm1           = update_energy ? (pmbp->phydro->peos->eos_data.gamma - 1.0) : 0.0;

  par_for(
    "enforce_isothermal_pressure",
    DevExeSpace(),
    0, (pmbp->nmb_thispack - 1),
    indcs.ks, indcs.ke,
    indcs.js, indcs.je,
    indcs.is, indcs.ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) 
  {
    const Real x1v = CellCenterX(i - indcs.is, indcs.nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real x2v = CellCenterX(j - indcs.js, indcs.nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    const Real x3v = CellCenterX(k - indcs.ks, indcs.nx3, size.d_view(m).x3min, size.d_view(m).x3max);

    const Real rho      = w(m, IDN, k, j, i);
    const Real cs2      = disk.SoundSpeedSquare(x1v, x2v, x3v, binary_position);
    const Real pressure = rho * cs2;
    w(m, IPR, k, j, i)  = pressure;

    if (update_energy) {
      const Real vx      = w(m, IVX, k, j, i);
      const Real vy      = w(m, IVY, k, j, i);
      const Real vz      = w(m, IVZ, k, j, i);
      const Real kinetic = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
      u(m, IEN, k, j, i) = pressure / gm1 + kinetic;
    }
  });
}



void enforce_floors(Mesh *pm) {

  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) {
    return;
  }

  auto &indcs      = pm->mb_indcs;
  auto &size       = pmbp->pmb->mb_size;
  auto &w          = pmbp->phydro->w0;
  auto &u          = pmbp->phydro->u0;
  const Real gamma = pmbp->phydro->peos->eos_data.gamma;
  const Real gm1   = gamma - 1.0;
  Real floor_dens  = g_floor_density;   // save globals to host for GPU scope
  Real floor_press = g_floor_pressure;  // save globals to host for GPU scope

  par_for(
    "enforce_floors",
    DevExeSpace(),
    0, (pmbp->nmb_thispack - 1),
    indcs.ks, indcs.ke,
    indcs.js, indcs.je,
    indcs.is, indcs.ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) 
  {

    bool update_cons = false;

    // Density
    Real dens = w(m, IDN, k, j, i);
    if (Kokkos::isnan(dens) || dens < floor_dens) {
      dens               = floor_dens;
      w(m, IDN, k, j, i) = dens;
      w(m, IVX, k, j, i) = 0.0;
      w(m, IVY, k, j, i) = 0.0;
      w(m, IVZ, k, j, i) = 0.0;
      update_cons = true;
    }

    // Pressure
    Real pres = w(m, IPR, k, j, i);
    if (Kokkos::isnan(pres) || pres < floor_press) {
      pres               = floor_press;
      w(m, IPR, k, j, i) = pres;
      update_cons = true;
    }

    // Check for NaNs in velocity and reset
    if (Kokkos::isnan(w(m, IVX, k, j, i))) w(m, IVX, k, j, i) = 0.0;
    if (Kokkos::isnan(w(m, IVY, k, j, i))) w(m, IVY, k, j, i) = 0.0;
    if (Kokkos::isnan(w(m, IVZ, k, j, i))) w(m, IVZ, k, j, i) = 0.0;

    // Update conserved variables if needed
    if (update_cons) {
      u(m, IDN, k, j, i) = dens;
      u(m, IM1, k, j, i) = w(m, IVX, k, j, i) * dens;
      u(m, IM2, k, j, i) = w(m, IVY, k, j, i) * dens;
      u(m, IM3, k, j, i) = w(m, IVZ, k, j, i) * dens;
      // Add kinetic energy to conservative energy
      Real kinetic = 0.5 * dens * (w(m, IVX, k, j, i) * w(m, IVX, k, j, i)
                                + w(m, IVY, k, j, i) * w(m, IVY, k, j, i)
                                + w(m, IVZ, k, j, i) * w(m, IVZ, k, j, i));
      u(m, IEN, k, j, i) = pres / gm1 + kinetic;
    }
  });
}




void CircumbinaryHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 18;
  pdata->label[0]  = "Mdot_1";     pdata->label[1]  = "Mdot_2";
  pdata->label[2]  = "Tx_grav_1";  pdata->label[3]  = "Tx_grav_2";
  pdata->label[4]  = "Ty_grav_1";  pdata->label[5]  = "Ty_grav_2";
  pdata->label[6]  = "Tz_grav_1";  pdata->label[7]  = "Tz_grav_2";
  pdata->label[8]  = "Tx_accr_1";  pdata->label[9]  = "Tx_accr_2";
  pdata->label[10] = "Ty_accr_1";  pdata->label[11] = "Ty_accr_2";
  pdata->label[12] = "Tz_accr_1";  pdata->label[13] = "Tz_accr_2";
  pdata->label[14] = "Pgrav_1";    pdata->label[15] = "Pgrav_2";
  pdata->label[16] = "Paccr_1";    pdata->label[17] = "Paccr_2";

  if (g_binary == nullptr) {
    for (int n = 0; n < pdata->nhist; ++n) { pdata->hdata[n] = 0.0; }
    return;
  }

  auto h_mass = Kokkos::create_mirror_view_and_copy(HostMemSpace(), g_mass_accreted);
  auto h_pg   = Kokkos::create_mirror_view_and_copy(HostMemSpace(), g_momentum_grav);
  auto h_pa   = Kokkos::create_mirror_view_and_copy(HostMemSpace(), g_momentum_accreted);
  auto h_eg   = Kokkos::create_mirror_view_and_copy(HostMemSpace(), g_energy_grav);
  auto h_ea   = Kokkos::create_mirror_view_and_copy(HostMemSpace(), g_energy_accreted);

  const auto pos = g_binary->BinaryPosition(pm->time);

  const Real interval = pm->time - g_last_reset_time;
  const Real inv_dt = (interval > 0.0) ? (1.0 / interval) : 0.0;

  for (int s = 0; s < 2; ++s) {
    const Real x0 = pos[s][0], y0 = pos[s][1], z0 = pos[s][2];

    pdata->hdata[0 + s]  = h_mass(s) * inv_dt;

    pdata->hdata[2 + s]  = (y0*h_pg(s,2) - z0*h_pg(s,1)) * inv_dt;  // Tx, gravity
    pdata->hdata[4 + s]  = (z0*h_pg(s,0) - x0*h_pg(s,2)) * inv_dt;  // Ty, gravity
    pdata->hdata[6 + s]  = (x0*h_pg(s,1) - y0*h_pg(s,0)) * inv_dt;  // Tz, gravity

    pdata->hdata[8 + s]  = (y0*h_pa(s,2) - z0*h_pa(s,1)) * inv_dt;  // Tx, accretion
    pdata->hdata[10 + s] = (z0*h_pa(s,0) - x0*h_pa(s,2)) * inv_dt;  // Ty, accretion
    pdata->hdata[12 + s] = (x0*h_pa(s,1) - y0*h_pa(s,0)) * inv_dt;  // Tz, accretion

    pdata->hdata[14 + s] = h_eg(s) * inv_dt;  // power delivered by gravity
    pdata->hdata[16 + s] = h_ea(s) * inv_dt;  // power delivered by accretion
  }

  // Reset for the next interval.
  Kokkos::deep_copy(g_mass_accreted, 0.0);
  Kokkos::deep_copy(g_momentum_grav, 0.0);
  Kokkos::deep_copy(g_momentum_accreted, 0.0);
  Kokkos::deep_copy(g_energy_grav, 0.0);
  Kokkos::deep_copy(g_energy_accreted, 0.0);
  g_last_reset_time = pm->time;
}


void CircumbinaryFinalAnalysis(ParameterInput *pin, Mesh *pm) {
  g_mass_accreted     = Kokkos::View<Real[2]>();
  g_momentum_grav     = Kokkos::View<Real[2][3]>();
  g_momentum_accreted = Kokkos::View<Real[2][3]>();
  g_energy_grav       = Kokkos::View<Real[2]>();
  g_energy_accreted   = Kokkos::View<Real[2]>();
}
