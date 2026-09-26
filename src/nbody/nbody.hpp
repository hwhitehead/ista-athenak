#ifndef NBODY_NBODY_HPP_
#define NBODY_NBODY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file nbody.hpp
//  \brief definitions for nbody class

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "diffusion/sts_types.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class EquationOfState;
class Coordinates;
class Viscosity;
class Conduction;
class SourceTerms;
class OrbitalAdvectionCC;
class ShearingBoxCC;
class Driver;

struct NBodyTaskIDs {
    TaskID reduce_mesh, reduce_meshes, integrate, scatter, calc_dt;
    TaskID initrk, flux, rkupdt;
};

// indices for principle nbody data register
// contains data read by source terms for gravity, accretion forcing
enum NBodyDataIndices {M_DATA = 0, 
                        X_DATA = 1, Y_DATA = 2, Z_DATA = 3,
                        VX_DATA = 4, VY_DATA = 5, VZ_DATA = 6,
                        AX_DATA = 7, AY_DATA = 8, AZ_DATA = 9,
                        R_SOFT_DATA = 10, R_AMR_DATA = 11, N_AMR_DATA = 12,
                        NVAR_DATA = 13};

// indices for history output writing
enum NBodyHistIndices {M_HIST = 0, 
                        X_HIST = 1, Y_HIST = 2, Z_HIST = 3,
                        VX_HIST = 4, VY_HIST = 5, VZ_HIST = 6,
                        AX_GRAV_HIST = 7, AY_GRAV_HIST = 8, AZ_GRAV_HIST = 9,
                        AX_ACC_HIST = 10, AY_ACC_HIST = 11, AZ_ACC_HIST = 12,
                        NVAR_HIST = 13};

// indices for backreaction registers
// after gather, register populated with derivates, not deltas (hence double definitions)
enum NBodyBackIndices {DM_BACK = 0, MDOT_BACK = 0,
                        DPX_GRAV_BACK = 1, AX_GRAV_BACK = 1, 
                        DPY_GRAV_BACK = 2, AY_GRAV_BACK = 2,
                        DPZ_GRAV_BACK = 3, AZ_GRAC_BACK = 3,
                        DPX_ACC_BACK = 4, AX_ACC_BACK = 4,
                        DPY_ACC_BACK = 5, AY_ACC_BACK = 5,
                        DPZ_ACC_BACK = 6, AZ_ACC_BACK = 6,
                        NVAR_BACK = 7};

// indicies for general registers
// contains data not related to a specific bodhy
enum NBodyGeneralIndices {DE_GENERAL = 0, 
                          NVAR_GENERAL = 1};

// indices for nbody integration registers (scratch)
enum NBodyRegisterIndices {M_REG = 0, MDOT_REG = 0, 
                            X_REG = 1, XDOT_REG = 1, 
                            Y_REG = 2, YDOT_REG = 2, 
                            Z_REG = 3, ZDOT_REG = 3,
                            VX_REG = 4, VXDOT_REG = 4, 
                            VY_REG = 5, VYDOT_REG = 5, 
                            VZ_REG = 6, VZDOT_REG = 6,
                            NVAR_REG = 7};

Real CalcLocalSoundSpeedSqr(DualArray2D<Real> nbody_data, int num_nbody, Real inv_Mach_sqr, const Real x, const Real y, const Real z);

namespace nbody {

//----------------------------------------------------------------------------------------
//! \fn  class NBody
//! \brief The NBody class tracks an arbitrary number of bodies that evolve in tandem
//! with the Hydro state. Initial conditions are specified in the athinput file using the 
//! <nbody> block. During runtime, the body masses, positions and velocites are evolved 
//! according to their mutual gravity and coupling with the gas, connected using source 
//! terms which allow for gas-body gravity and accretion. The nbody state is evolved on 
//! the same timestep as the Hydro state, ensuring direct 
class NBody {
  public:
    NBody(MeshBlockPack *ppack, ParameterInput *pin);
    ~NBody();
    
    // Data
    int num_nbody;                // number of discrete bodies to track
    Real dt_new, dt_old;          // stable nbody timestep
    Real eta_dt;                  // prefactor for stable timestep
    DualArray2D<Real> nbody_data; // principle data register shape = (num_nbody, NVAR_DATA)
    bool verbose;                 // boolean flag for command line progress writes

    // Back reaction communicators shape = (num_nbody, NVAR_REG)
    DualArray2D<Real> delta_this_pack;  // DUAL summation for backreaction in this MeshBlockPack
    HostArray2D<Real> delta_this_mesh;  // HOST summation for backreaction in this Mesh (this rank)
    HostArray2D<Real> delta_all_meshes; // HOST summation for backreaction in all Meshes (all ranks)

    // Registers used for time evolution shape = (num_nbody, NVAR_REG)
    HostArray2D<Real> nbody_data1;  // nbody state at intermediate time step
    HostArray2D<Real> nbody_flux;   // time derivative of current nbody state

    // WIP: DUAL register for non body specific quantities shape = (NVAR_GEN)
    DualArray2D<Real> general_data; 
    
    // container to hold names of TaskIDs
    NBodyTaskIDs id;

    // physics module booleans
    bool src_gravity, src_local_iso, src_accretion;   // Hydro toggles
    bool inc_backreaction, sum_backreaction, inc_pn;  // NBody toggles

    // disc state variables (for sound speed compute)
    Real Mach, inv_Mach_sqr;

    // physical units (for post-newtonian terms)
    Real unit_L, unit_M, unit_T;  // set by user in athinput
    Real unit_V, unit_A;          // compound units (derived)

    // task functions
    void AssembleNBodyTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
    TaskStatus InitRK(Driver *d, int state);        // prep intermediate register
    TaskStatus Fluxes(Driver *pdrive, int stage);   // compute derivate of nbody state
    TaskStatus RKUpdate(Driver *pdrive, int stage); // propogate nbody state
    // TaskStatus ReduceParentMesh(Driver *d, int state);
    // TaskStatus ReduceAllMeshes(Driver *d, int state);
    // TaskStatus Integrate(Driver *d, int stage);
    // TaskStatus Scatter(Driver *d, int state);
    // TaskStatus NewTimeStep(Driver *pdrive, int stage);

    // non-task methods
    Real CalcTimeStep();
    void EvaluateF(DualArray2D<Real> y, DualArray2D<Real> &f);
    void NBodySrcTerms(const Real beta_dt);         // wrapped to call all source terms
    void NBodyPointSrcTerm(const Real beta_dt);     // per-body source terms (gravity, accretion)
    void NBodyIsoSrcTerm(const Real beta_dt);       // enforce local isothermality
    Real CalcLocalOmegaSqr(const Real x, const Real y, const Real z); // TODO: deprecated for new cs method (in diff.)

  private:
    MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this NBody
    Real _G = 1.0;            // gravitational constant (TODO: deprivatise)
    // these registers are only used for coarse timestep RK4 integration, depreciate 
    DualArray2D<Real> _y_init, _y_sub, _y_ret; 
    DualArray2D<Real> _k_sub;
}; // end NBody class

} // end namespace nbody

#endif // NBODY_NBODY_HPP_