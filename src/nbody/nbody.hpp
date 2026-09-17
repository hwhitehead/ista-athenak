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
    TaskID gather, integrate, scatter, calc_dt;
};

enum NBodyDataIndices {M_DATA = 0, 
                        X_DATA = 1, Y_DATA = 2, Z_DATA = 3,
                        VX_DATA = 4, VY_DATA = 5, VZ_DATA = 6,
                        AX_DATA = 7, AY_DATA = 8, AZ_DATA = 9
                        R_SOFT_DATA = 10, 
                        NVAR_DATA = 11};

enum NBodyHistIndices {M_HIST = 0, 
                        X_HIST = 1, Y_HIST = 2, Z_HIST = 3,
                        VX_HIST = 4, VY_HIST = 5, VZ_HIST = 6
                        NVAR_HIST = 7};

enum NBodyBackIndices {DM_BACK = 0, DVX_BACK = 1, DVY_BACK = 2, DVZ_BACK = 3
                        NVAR_BACK = 4};

enum NBodyRegisterIndices {MDOT_REG = 0, 
                            XDOT_REG = 1, YDOT_REG = 2, ZDOT_REG = 3,
                            VXDOT_REG = 4, VYDOT_REG = 5, VZDOT_REG = 6
                            NVAR_REG = 7};

namespace nbody {

class NBody {
  public:
    NBody(MeshBlockPack *ppack, ParameterInput *pin);
    ~NBody();
    
    int num_nbody; // number of discrete particles to track
    int var_per_body; // (m,3x,3vx,+extras)
    Real dt_new, dt_old; // nbody timestep (before prefactor scaling)
    Real eta_dt; // prefactor for timestep scaling

    // principle data register for wider access 
    // shape (num_nbody, var_per_body)
    DualArray2D<Real> nbody_data; 

    // summation space for mb_pack level updates
    DualArray2D<Real> delta_nbody_data;

    // container to hold names of TaskIDs
    NbodyTaskIDs id;

    // physics module booleans
    bool src_gravity, src_accretion;

    // task functions
    void AssembleNBodyTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
    TaskStatus Gather(Driver *d, int state);
    TaskStatus Integrate(Driver *d, int stage);
    TaskStatus Scatter(Driver *d, int state);
    TaskStatus NewTimeStep(Driver *pdrive, int stage);

    // methods
    Real CalcTimeStep();
    void EvaluateF(DualArray2D<Real> y, DualArray2D<Real> &f);
    void NBodySrcTerms(const Real beta_dt);
    void NBodyGravitySrcTerm(const Real beta_dt);

  private:
    MeshBlockPack* pmy_pack;
    int _reg_per_body = 7; // (m,3x,3vx)
    // private registers for intermediate integrator states (rk4)
    DualArray2D<Real> _y_init, _y_sub, _y_ret; 
    DualArray2D<Real> _k_sub;
    Real _G = 1.0; // gravitational constant
};

} // end namespace nbody

#endif // NBODY_NBODY_HPP_