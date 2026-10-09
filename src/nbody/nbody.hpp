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
#include "eos/eos.hpp"

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
    TaskID initrk, flux, rkupdt, send, newdt;
};

// indices for principle nbody data register
// contains data read by source terms for gravity, accretion forcing
enum NBodyDataIndices {M_DATA = 0, 
                        X_DATA = 1, Y_DATA = 2, Z_DATA = 3,
                        VX_DATA = 4, VY_DATA = 5, VZ_DATA = 6, MDOT_DATA = 7,
                        AX_GRAV_DATA = 8, AY_GRAV_DATA = 9, AZ_GRAV_DATA = 10,
                        AX_ACC_DATA = 11, AY_ACC_DATA = 12, AZ_ACC_DATA = 13,
                        R_SOFT_DATA = 14, R_AMR_DATA = 15, T_AMR_DATA = 16,
                        NVAR_DATA = 17};

// indices for history output writing
enum NBodyHistIndices {M_HIST = 0, 
                        X_HIST = 1, Y_HIST = 2, Z_HIST = 3,
                        VX_HIST = 4, VY_HIST = 5, VZ_HIST = 6, MDOT_HIST = 7,
                        AX_GRAV_HIST = 8, AY_GRAV_HIST = 9, AZ_GRAV_HIST = 10,
                        AX_ACC_HIST = 11, AY_ACC_HIST = 12, AZ_ACC_HIST = 13,
                        NVAR_HIST = 14};

// indices for backreaction registers
// after gather, register populated with derivates, not deltas (hence double definitions)
enum NBodyBackIndices {DM_BACK = 0, MDOT_BACK = 0,
                        DPX_GRAV_BACK = 1, AX_GRAV_BACK = 1, 
                        DPY_GRAV_BACK = 2, AY_GRAV_BACK = 2,
                        DPZ_GRAV_BACK = 3, AZ_GRAV_BACK = 3,
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

namespace nbody {

void NBodyHistory(HistoryData *pdata, Mesh *pm);
void NBodyTrackRefinementCondition(MeshBlockPack* pmbp);

//----------------------------------------------------------------------------------------
//! \fn  class NBody
//! \brief The NBody class tracks an arbitrary number of bodies that evolve in tandem
//! with the Hydro state. Initial conditions are specified in the athinput file using the 
//! <nbody> block. During runtime, the body masses, positions and velocites are evolved 
//! according to their mutual gravity and coupling with the gas, connected using source 
//! terms which allow for gas-body gravity and accretion. The nbody state is evolved on 
//! the same timestep as the Hydro state, ensuring direct synchronicity
class NBody {
  public:
    NBody(MeshBlockPack *ppack, ParameterInput *pin);
    ~NBody();
    
    // Data
    int num_nbody;                // number of discrete bodies to track
    Real dt_new, dt_old;          // stable nbody timestep
    Real dt_visc;                 // stable timestep for diffusive module
    Real eta_dt;                  // prefactor for stable timestep
    Real G_const;                 // gravitational constant in code units
    DualArray2D<Real> nbody_data; // principle data register shape = (num_nbody, NVAR_DATA)
    bool verbose;                 // boolean flag for command line progress writes
    Real rho_sink_floor;          // minimum density in sink region 

    // Back reaction communicators shape = (num_nbody, NVAR_REG)
    DualArray2D<Real> delta_this_pack;  // DUAL summation for backreaction in this MeshBlockPack
    HostArray2D<Real> delta_this_mesh;  // HOST summation for backreaction in this Mesh (this rank)
    HostArray2D<Real> delta_all_meshes; // HOST summation for backreaction in all Meshes (all ranks)

    // Registers used for time evolution shape = (num_nbody, NVAR_REG)
    HostArray2D<Real> nbody_data1;  // nbody state at intermediate time step
    HostArray2D<Real> nbody_flux;   // time derivative of current nbody state
    
    // MPI communication buffer shape = (num_nbody, NVAR_DATA)
    HostArray2D<Real> nbody_comm;   // buffer for MPI communication in NBody::Send

    // WIP: DUAL register for non body specific quantities shape = (NVAR_GEN)
    DualArray2D<Real> general_data; 
    
    // container to hold names of TaskIDs
    NBodyTaskIDs id;

    // physics module booleans
    bool src_gravity, src_accretion;                    // forcing toggles
    bool src_local_iso, src_blackbody, src_beta_cool;   // thermodynamic toggles
    bool inc_backreaction, sum_backreaction, inc_pn;    // NBody toggles
    int  sink_mode;                                     // accretion routine (0 = fixed rate, 1 = flat visc, 2 = dyn visc)

    // disc state variables (for sound speed compute)
    Real Mach, inv_Mach_sqr, cs_sqr_floor;

    // physical units (for post-newtonian terms)
    Real unit_L, unit_M, unit_T;  // set by user in athinput
    Real unit_V, unit_A;          // compound units (derived)
    Real kappa_es_code;           // electron scattering opacity in code units  
    Real rad_const_code;          // radiation constant in code units 
    Real c_light_code;            // speed of light in code units 
    Real k_boltzmann_code;        // boltzmann constant in code units
    Real mu_gas;                  // mean molecular weight of gas in code units
    Real proton_mass_code;        // proton mass in code units
    Real gas_const_code;          // gas constant in code units

    // task functions
    void AssembleNBodyTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
    TaskStatus InitRK(Driver *d, int state);            // prep intermediate register
    TaskStatus Fluxes(Driver *pdrive, int stage);       // compute derivate of nbody state
    TaskStatus RKUpdate(Driver *pdrive, int stage);     // propogate nbody state
    TaskStatus Send(Driver *pdrive, int stage);     // pass updated state to all NBody instances
    TaskStatus NewTimeStep(Driver *pdrive, int stage);  // compute next stable time step

    // non-task methods
    Real CalcTimeStep();
    void EvaluateF(DualArray2D<Real> y, DualArray2D<Real> &f);
    void NBodySrcTerms(const Real beta_dt);         // wrapped to call all source terms
    void NBodyPointSrcTerm(const Real beta_dt);     // per-body source terms (gravity, accretion)
    void NBodyForcedIsoSrcTerm(const Real beta_dt);       // enforce local isothermality
    void NBodyBlackBodySrcTerm(const Real beta_dt); // cool as black body 
    void NBodyBetaCoolSrcTerm(const Real beta_dt);  // beta cooling

    // inhomogeneous viscosity handling
    Real alpha; // viscosity coefficient for alpha-disc prescription
    void CalcViscousFluxAlpha(const DvceArray5D<Real> &w0); // compute by-cell viscosity according to alpha prescription
    void AddViscousFlux(const DvceArray5D<Real> &w0, const EOS_Data &eos, DvceFaceFld5D<Real> &flx);
    void NewViscousTimeStep(const DvceArray5D<Real> &w, const EOS_Data &eos_data);
    DvceArray4D<Real> nu_iso;      // inhomogeneous viscosity coefficient, shape = (nmb, k, j, i)
    DualArray1D<Real> max_nu_iso;  // maximum viscosity coefficient for each MeshBlock, shape = (nmb)

  private:
    MeshBlockPack* pmy_pack;  // ptr to MeshBlockPack containing this NBody
}; // end NBody class

} // end namespace nbody

#endif // NBODY_NBODY_HPP_