#ifndef  CURVED_SURFACE_HEADER_H
#define  CURVED_SURFACE_HEADER_H

// ------------------------------------

#define  DEF_BLOCK_SIZE     128

#define  MAX_NUM_OBJ_PARAMS   4   


#ifdef USE_DOUBLE
#define  real   double
#define  real2  double2
#define  real4  double4
#define  make_real2  make_double2
#define  make_real4  make_double4
#define  rSqrt  sqrt
#define  rSin   sin
#define  rCos   cos
#define  REAL_ZERO  0.0
#define  REAL_ONE   1.0
#define  REAL_TWO   2.0
#define  REAL_HALF  0.5
#define  REAL_SIX   6.0
#define  REAL_EPS   1e-4
#define  PI         3.141592653589793
#define  TWO_PI     6.283185307179586

#define  CU_KAPPA    253.27
#define  Q_BO_M      5.39957766969e6  //  e*B0/m_el   electron charge * earth magnetic field / electron mass
#define  C_RE       47.003138691     // = speed of light / earth radius

#else // USE_DOUBLE

#define  real   float
#define  real2  float2
#define  real4  float4
#define  make_real2  make_float2
#define  make_real4  make_float4
#define  rSqrt  sqrtf
#define  rSin   sinf
#define  rCos   cosf
#define  REAL_ZERO  0.0f
#define  REAL_ONE   1.0f
#define  REAL_TWO   2.0f
#define  REAL_HALF  0.5f
#define  REAL_SIX   6.0f
#define  REAL_EPS   0.0001f
#define  PI         3.14159f
#define  TWO_PI     6.28319f

#define  CU_KAPPA    253.27f
#define  Q_BO_M      5.39958e6
#define  C_RE       47.00314

#endif // USE_DOUBLE

#define  SQR(x) ((x)*(x))

typedef unsigned int uint;

// Simulation parameters
typedef struct CSParams_t
{
   int    numParticles;    //!< Total number of particles
   real   damp;            //!< Damping factor
   real   hStep;           //!< Step size for integration
   real   velReflDamp;     //!< Velocity reflection damping 
   
   real   E[3];            //!< Constant electric field
   real   B[3];            //!< Constant magnetic field
} CSparams;

// Object parameters
typedef struct CSObjParams_t
{
   int  obj_type;          //!< Object type: sphere, torus,...
   real center[3];         //!< Center of the object's coord. system
   real e1[3];             //!< Base vector 1 of object's coord. system
   real e2[3];             //!< Base vector 2 of object's coord. system
   real e3[3];             //!< Base vector 3 of object's coord. system
   real u_range[2];        //!< Domain of parameter u.
   real v_range[2];        //!< Domain of parameter v.
   real u_mod;             //!< Modulo of parameter u.
   real v_mod;             //!< Modulo of parameter v.
   int  use_modulo[2];     //!< Use modulo or not.
   real value[MAX_NUM_OBJ_PARAMS];  //!< object parameters
} CSObjParams;

#endif // _CURVED_SURFACE_HEADER_H
