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

struct NbodyTaskIDs {
    TaskID integrate;
};

namespace nbody {

class NBody {
  public:
    NBody(MeshBlockPack *ppack, ParameterInput *pin);
    ~NBody();
    
    // data
    int num_nbody; // number of discrete particles to trakc
    // principle data register for wider access 
    // shape (num_nbody, _var_per_body)
    DualArray2D<Real> nbody_data; 

    // container to hold names of TaskIDs
    NbodyTaskIDs id;

    // task functions
    void AssembleNBodyTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
    TaskStatus Integrate(Driver *d, int stage);
    TaskStatus Communicate(Driver *d, int state);
    void EvaluateF(DvceArray1D<Real> y, DvceArray1D<Real> &f);


  private:
    MeshBlockPack* pmy_pack;
    int _var_per_body = 7; // (m,x,y,z,vx,vy,vz)
    // private registers for intermediate integrator states (rk4)
    // all private registers are len 7 * num_nbody (m,x,y,z,vx,vy,vz)
    DualArray1D<Real> _y_init, _y_sub, _y_ret; 
    DualArray1D<Real> _k_sub;
};

} // end namespace nbody

#endif // NBODY_NBODY_HPP_