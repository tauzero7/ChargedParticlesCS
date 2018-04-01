#ifndef CURVED_SURFACE_CODE
#define CURVED_SURFACE_CODE

#include <cstdio>
#include <cstdlib>
#include "header.cuh"

// Implementation files for the diverse surface types...
#include "cs_plane.inl"
#include "cs_sphere.inl"
//#include "cs_ellipsoid.inl"
#include "cs_frustum.inl"
#include "cs_torus.inl"
#include "cs_moebius.inl"
#include "cs_graph.inl"


// These identifiers must be equal to e_object_type in 'defs.h' !!
#define   SURF_TYPE_PLANE         0
#define   SURF_TYPE_SPHERE        1
#define   SURF_TYPE_ELLIPSOID     2
#define   SURF_TYPE_FRUSTUM       3
#define   SURF_TYPE_TORUS         4
#define   SURF_TYPE_MOEBIUS       5
#define   SURF_TYPE_GRAPH         6

__constant__ real EPS_QUAD = (real)0.001;
//__constant__ real EPS_QUAD = (real)0;


/** Calculate surface coordinates from uv-parameters.
 * \param surfaceType : type of surface
 * \param params : object parameters
 * \param y : uv parameters
 * \param f : surface coordinates
 */
__host__ __device__  void  calcSurface( int surfaceType, CSObjParams* params, real* y, real* f )
{
   switch (surfaceType)
   {
      case SURF_TYPE_PLANE: {
         plane_calc_f(y,f,params);
         break;
      }
      case SURF_TYPE_SPHERE: {
         sphere_calc_f(y,f,params);
         break;
      }
      case SURF_TYPE_ELLIPSOID: {
         break;
      }
      case SURF_TYPE_FRUSTUM: {
         frustum_calc_f(y,f,params);
         break;
      }
      case SURF_TYPE_TORUS: {
         torus_calc_f(y,f,params);
         break;
      }
      case SURF_TYPE_MOEBIUS: {
         moebius_calc_f(y,f,params);
         break;
      }  
      case SURF_TYPE_GRAPH: {
         graph_calc_f(y,f,params);
         break;
      }
   }
}

/** Calculate surface normal at uv-parameters.
 * \param surfaceType : type of surface
 * \param params : object parameters
 * \param y : uv parameters
 * \param n : surface normal in 3D
 */
__host__ __device__  void  calcNormal( int surfaceType, CSObjParams* params, real* y, real* n )
{
   switch (surfaceType)
   {
      case SURF_TYPE_PLANE: {
         plane_calc_normal(y,n,params);
         break;
      }
      case SURF_TYPE_SPHERE: {
         sphere_calc_normal(y,n,params);
         break;
      }
      case SURF_TYPE_ELLIPSOID: {
         break;
      }
      case SURF_TYPE_FRUSTUM: {
         frustum_calc_normal(y,n,params);
         break;
      }
      case SURF_TYPE_TORUS: {
         torus_calc_normal(y,n,params);
         break;
      }
      case SURF_TYPE_MOEBIUS: {
         moebius_calc_normal(y,n,params);
         break;
      }  
      case SURF_TYPE_GRAPH: {
         graph_calc_normal(y,n,params);
         break;
      }
   }
}

/** Cross product
 * \param a
 * \param b
 * \param c = a times b
 */
__device__ void  crossProd( real* a, real* b, real* c )
{
   c[0] = a[1]*b[2] - a[2]*b[1];
   c[1] = a[2]*b[0] - a[0]*b[2];
   c[2] = a[0]*b[1] - a[1]*b[0];
}

/** Calculate Coulomb interaction
 * \param objNum : object number.
 * \param idx
 * \param y
 * \param dydt
 * \param oldPos
 * \param oldDir
 * \param mass
 * \param charge
 * \param params : simulation parameters.
 * \param allObjParams : object parameters of all objects.
 */
__device__ void  calcDerivs( int objNum, uint idx,
                             real* y, real* dydt,
                             real4* oldPos, real2* oldDir,
                             real* mass, real* charge, CSparams* params, CSObjParams* allObjParams )
{
   dydt[0] = y[2];
   dydt[1] = y[3];
   dydt[2] = REAL_ZERO;
   dydt[3] = REAL_ZERO;

   real christoffel[8];

   real fi[3];
   real dfdu1i[3];
   real dfdu2i[3];
   real invmetric[4];

   CSObjParams* objParams = &allObjParams[objNum];

   // -------------------------------------
   //   Curved surface
   // -------------------------------------

   // see 'e_object_type' numeration in 'defs.h'
   int surfaceType = allObjParams[objNum].obj_type;
   switch (surfaceType)
   {
      case SURF_TYPE_PLANE: {
         plane_calc_chris     ( y,christoffel,objParams );
         plane_calc_f         ( y,fi,objParams );
         plane_calc_df        ( y,dfdu1i,dfdu2i,objParams );
         plane_calc_invMetric ( y,invmetric,objParams );
         break;
      }
      case SURF_TYPE_SPHERE: {
         sphere_calc_chris     ( y,christoffel,objParams );
         sphere_calc_f         ( y,fi,objParams );
         sphere_calc_df        ( y,dfdu1i,dfdu2i,objParams );
         sphere_calc_invMetric ( y,invmetric,objParams );
         break;
      }
      case SURF_TYPE_ELLIPSOID: {
         break;
      }
      case SURF_TYPE_FRUSTUM: {
         frustum_calc_chris     ( y,christoffel,objParams );
         frustum_calc_f         ( y,fi,objParams );
         frustum_calc_df        ( y,dfdu1i,dfdu2i,objParams );
         frustum_calc_invMetric ( y,invmetric,objParams );
         break;
      }
      case SURF_TYPE_TORUS: {
         torus_calc_chris     ( y,christoffel,objParams );
         torus_calc_f         ( y,fi,objParams );
         torus_calc_df        ( y,dfdu1i,dfdu2i,objParams );
         torus_calc_invMetric ( y,invmetric,objParams );
         break;
      }
      case SURF_TYPE_MOEBIUS: {
         moebius_calc_chris     ( y,christoffel,objParams );
         moebius_calc_f         ( y,fi,objParams );
         moebius_calc_df        ( y,dfdu1i,dfdu2i,objParams );
         moebius_calc_invMetric ( y,invmetric,objParams );
         break;
      }  
      case SURF_TYPE_GRAPH: {
         graph_calc_chris     ( y,christoffel,objParams );
         graph_calc_f         ( y,fi,objParams );
         graph_calc_df        ( y,dfdu1i,dfdu2i,objParams );
         graph_calc_invMetric ( y,invmetric,objParams );
         break;
      }
   }

   int num2,num3;
   for(int i=0; i<2; i++) {
      for(int j=0; j<2; j++) {
         num2 = 4*0 + i*2 + j;  // christoffel[i][j][0]
         num3 = 4*1 + i*2 + j;  // christoffel[i][j][1]
         dydt[2] -= christoffel[num2]*y[2+i]*y[2+j];
         dydt[3] -= christoffel[num3]*y[2+i]*y[2+j];
      }
   }
   

#if 1
   // -------------------------------------
   //   Charged particle interaction
   // -------------------------------------
   real fp[3];
   real diff[3];
   real yp[2];
   
   real a1 = REAL_ZERO;
   real a2 = REAL_ZERO;

   real chargeQ = charge[idx];
   real oOmassQ = REAL_ONE/mass[idx];
   real chargeP;
   unsigned int oldObjNum;
   for(int p=0; p<params->numParticles; p++)
   {
      if (p!=idx)
      {
         yp[0] = oldPos[p].x;
         yp[1] = oldPos[p].y;
         oldObjNum = (unsigned int)oldPos[p].w;
         CSObjParams* pObjParams = &allObjParams[oldObjNum];

         // calculate position of the p-particle
         calcSurface(surfaceType,pObjParams,yp,fp);

         diff[0] = fi[0] - fp[0];
         diff[1] = fi[1] - fp[1];
         diff[2] = fi[2] - fp[2];

         // Plummer-potential
         real r = rSqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2] + EPS_QUAD);
         real edr = REAL_ONE/(r*r*r);

         chargeP = charge[p];
         real br1 = (diff[0]*dfdu1i[0] + diff[1]*dfdu1i[1] + diff[2]*dfdu1i[2])*edr;
         real br2 = (diff[0]*dfdu2i[0] + diff[1]*dfdu2i[1] + diff[2]*dfdu2i[2])*edr;
         a1 += chargeQ*chargeP*(br1*invmetric[0] + br2*invmetric[2]);
         a2 += chargeQ*chargeP*(br1*invmetric[1] + br2*invmetric[3]);
      }
   }
   
   dydt[2] += CU_KAPPA*oOmassQ*a1;
   dydt[3] += CU_KAPPA*oOmassQ*a2;
#endif   


#if 1
   // -------------------------------------
   //   external electric field
   // -------------------------------------
   real Edfdu1 = params->E[0]*dfdu1i[0] + params->E[1]*dfdu1i[1] + params->E[2]*dfdu1i[2];
   real Edfdu2 = params->E[0]*dfdu2i[0] + params->E[1]*dfdu2i[1] + params->E[2]*dfdu2i[2];
   
   dydt[2] += chargeQ*oOmassQ*(Edfdu1*invmetric[0] + Edfdu2*invmetric[2]);
   dydt[3] += chargeQ*oOmassQ*(Edfdu1*invmetric[1] + Edfdu2*invmetric[3]);

   // -------------------------------------
   //   external magnetic field
   // -------------------------------------
   real df_cross_B1[3];
   real df_cross_B2[3];
   // -------------------------------------
   //   \dot{\vec{f}}\times\vec{B} = 
   //       \sum \dot{u}^n (\partial\vec{f}/\partial u^n) \times \vec{B}
   // -------------------------------------
   crossProd(dfdu1i,params->B,df_cross_B1);
   crossProd(dfdu2i,params->B,df_cross_B2);
   
   real df_cross_B[3];
   df_cross_B[0] = y[2]*df_cross_B1[0] + y[3]*df_cross_B2[0];
   df_cross_B[1] = y[2]*df_cross_B1[1] + y[3]*df_cross_B2[1];
   df_cross_B[2] = y[2]*df_cross_B1[2] + y[3]*df_cross_B2[2];
   
   // \dot{\vec{f}}\times\vec{B} * \partial\vec{f}/\partial u^j
   real dfcB_dfdu1 = df_cross_B[0]*dfdu1i[0] + df_cross_B[1]*dfdu1i[1] + df_cross_B[2]*dfdu1i[2];
   real dfcB_dfdu2 = df_cross_B[0]*dfdu2i[0] + df_cross_B[1]*dfdu2i[1] + df_cross_B[2]*dfdu2i[2];
   
   dydt[2] += chargeQ*oOmassQ*(dfcB_dfdu1*invmetric[0] + dfcB_dfdu2*invmetric[2]);   
   dydt[3] += chargeQ*oOmassQ*(dfcB_dfdu1*invmetric[1] + dfcB_dfdu2*invmetric[3]);
#endif

   // -------------------------------------
   //   friction
   // -------------------------------------
   dydt[2] -= params->damp*y[2];
   dydt[3] -= params->damp*y[3];   
}


/**  Standard fourth order Runge-Kutta integrator
 * \param objNum : object number.
 * \param idx
 * \param yo
 * \param yn
 * \param oldPOs
 * \param oldDir
 * \param mass
 * \param charge
 * \param params : simulation parameters.
 * \param allObjParams : object parameters of all objects.
 */
__device__ void  RK4 ( int objNum, uint idx,
                       real* yo, real* yn,
                       real4* oldPos, real2* oldDir,
                       real* mass, real* charge, CSparams* params, CSObjParams* allObjParams )
{
   real dydx[4];
   real k1[4];
   real k2[4];
   real k3[4];
   real k4[4];
   real yy[4];

   real h = params->hStep;

   calcDerivs( objNum, idx, yo, dydx, oldPos, oldDir, mass, charge, params, allObjParams );
   for(int i=0; i<4; i++)
   {
      k1[i] = h*dydx[i];
      yy[i] = yo[i] + REAL_HALF*k1[i];
   }

   calcDerivs( objNum, idx, yy, dydx, oldPos, oldDir, mass, charge, params, allObjParams );
   for(int i=0; i<4; i++)
   {
      k2[i] = h*dydx[i];
      yy[i] = yo[i] + REAL_HALF*k2[i];
   }

   calcDerivs( objNum, idx, yy, dydx, oldPos, oldDir, mass, charge, params, allObjParams );
   for(int i=0; i<4; i++)
   {
      k3[i] = h*dydx[i];
      yy[i] = yo[i] + k3[i];
   }

   calcDerivs( objNum, idx, yy, dydx, oldPos, oldDir, mass, charge, params, allObjParams );
   for(int i=0; i<4; i++)
   {
      k4[i] = h*dydx[i];
      yn[i] = yo[i] + REAL_ONE/REAL_SIX*(k1[i] + REAL_TWO*k2[i] + REAL_TWO*k3[i] + k4[i]);
   }
}

/**  Standard second order Runge-Kutta integrator
 * \param objNum : object number.
 * \param idx
 * \param yo
 * \param yn
 * \param oldPOs
 * \param oldDir
 * \param mass
 * \param charge
 * \param params : simulation parameters.
 * \param allObjParams : object parameters of all objects.
 */
__device__ void  RK2 ( int objNum, uint idx,
                       real* yo, real* yn,
                       real4* oldPos, real2* oldDir,
                       real* mass, real* charge, CSparams* params, CSObjParams* allObjParams )
{
   real dydx[4];
   real k1[4];
   real k2[4];
   real yy[4];

   real h = params->hStep;

   calcDerivs( objNum, idx, yo, dydx, oldPos, oldDir, mass, charge, params, allObjParams );
   for(int i=0; i<4; i++)
   {
      k1[i] = h*dydx[i];
      yy[i] = yo[i] + k1[i];
   }

   calcDerivs( objNum, idx, yy, dydx, oldPos, oldDir, mass, charge, params, allObjParams );
   for(int i=0; i<4; i++)
   {
      k2[i] = h*dydx[i];
      yn[i] = yo[i] + REAL_HALF*(k1[i]+k2[i]);
   }
}


/** Self-defined modulo function.
 * \param x
 * \param y
 */
__device__ real  mmod( real x, real y )
{
   if (y==(real)0)
      return x;
   return x - floor(x/y)*y;
}



/** Calculate time step for curved surface.
 * \param params : simulation parameters.
 * \param allObjParams : object parameters of all objects.
 * \param currPos : current positions
 * \param currVel : current velocities
 * \param currCol : current colors
 * \param newPos : new positions
 * \param newVel : new velocities
 * \param newCol : new colors
 * \param mass
 * \param charge
 */
__global__ void  curvedSurfaceStep( unsigned int numParticles,
                                    CSparams* params, CSObjParams* allObjParams,
                                    real4* currPos, real2* currVel, float4* currCol,
                                    real4* newPos,  real2* newVel,  float4* newCol,
                                    real* mass, real* charge )
{
   uint idx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
   if (idx>=numParticles)
      return;

   real y[4],ytemp[4];
   ytemp[0] = y[0] = currPos[idx].x;
   ytemp[1] = y[1] = currPos[idx].y;
   ytemp[2] = y[2] = currVel[idx].x;
   ytemp[3] = y[3] = currVel[idx].y;

   // particle belongs to surface with object number 'objNum'
   unsigned int objNum  = (unsigned int)currPos[idx].w;

   // if isFixed==1 then the particle is fixed at its position
   int isFixed = (int)currPos[idx].z;

   CSObjParams* particleObj = &allObjParams[objNum];
   if (isFixed!=1)
   {      
   #if 0
      // +++++ Runge-Kutta fourth order +++++     
      RK4(objNum,idx,y,ytemp,currPos,currVel,mass,charge,params,allObjParams);
   #else      
      // +++++ Runge-Kutta second order +++++
      RK2(objNum,idx,y,ytemp,currPos,currVel,mass,charge,params,allObjParams);
   #endif      
      
      // if a particle hits the boundary of the domain, the velocity
      // is reversed and scaled by the factor  velReflDamp...      
      
      if (particleObj->use_modulo[0]==1) {
         ytemp[0] = mmod(ytemp[0]-particleObj->u_range[0],particleObj->u_mod);
      }
      else {
         if (ytemp[0] <= particleObj->u_range[0]) {
            ytemp[0] = particleObj->u_range[0];
            ytemp[2] = -ytemp[2]*params->velReflDamp;
         }
         if (ytemp[0] >= particleObj->u_range[1]) {
            ytemp[0] = particleObj->u_range[1];
            ytemp[2] = -ytemp[2]*params->velReflDamp;
         }
      }

      if (particleObj->use_modulo[1]==1) {
         ytemp[1] = mmod(ytemp[1]-particleObj->v_range[0],particleObj->v_mod);
      }
      else {
         if (ytemp[1] <= particleObj->v_range[0]) {
            ytemp[1] = particleObj->v_range[0];
            ytemp[3] = -ytemp[3]*params->velReflDamp;
         }
         if (ytemp[1] >= particleObj->v_range[1]) {
            ytemp[1] = particleObj->v_range[1];
            ytemp[3] = -ytemp[3]*params->velReflDamp;
         }
      }
   }

   newPos[idx] = make_real4(ytemp[0],ytemp[1],currPos[idx].z,currPos[idx].w);
   newVel[idx] = make_real2(ytemp[2],ytemp[3]);
   newCol[idx] = make_float4(currCol[idx].x,currCol[idx].y,currCol[idx].z,currCol[idx].w);
}



/** Calculate positions in R3 for all particles.
 * \param allObjParams : all object parameters
 * \param uvPos : pointer to current uv coordinates
 * \param cartPos  : pointer to current 3d Cartesian coordinates
 */
__global__ void calc_f_forall( unsigned int numParticles, 
                               CSObjParams* allObjParams,
                               real4* uvPos, real4* cartPos )
{
   uint idx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
   if (idx>=numParticles)
      return;
      
   unsigned int objNum = (unsigned int)uvPos[idx].w;
   int surfaceType = allObjParams[objNum].obj_type;

   real yp[2];
   real fp[3];
   yp[0] = uvPos[idx].x;
   yp[1] = uvPos[idx].y;

   CSObjParams* objParams = &allObjParams[objNum];

   // calculate position of the p-particle
   switch (surfaceType)
   {
      case SURF_TYPE_PLANE: {
         plane_calc_f(yp,fp,objParams);
         break;
      }
      case SURF_TYPE_SPHERE: {
         sphere_calc_f(yp,fp,objParams);
         break;
      }
      case SURF_TYPE_ELLIPSOID: {
         break;
      }
      case SURF_TYPE_FRUSTUM: {
         frustum_calc_f(yp,fp,objParams);
         break;
      }
      case SURF_TYPE_TORUS: {
         torus_calc_f(yp,fp,objParams);
         break;
      }
      case SURF_TYPE_MOEBIUS: {
         moebius_calc_f(yp,fp,objParams);
         break;
      }  
      case SURF_TYPE_GRAPH: {
         graph_calc_f(yp,fp,objParams);
         break;
      }
   }
   cartPos[idx] = make_real4(fp[0],fp[1],fp[2],REAL_ONE);
}



/** Calculate velocities in R3 for all particles,  dot(f)^2
 * \param allObjParams : all object parameters
 * \param uvPos : pointer to current uv coordinates
 * \param uvVel : pointer to current uv velocities
 * \param cartVelSqr : pointer to current 3d Cartesian velocities
 */
__global__ void calc_fdfsqr_forall( unsigned int numParticles, CSObjParams* allObjParams,
                                    real4* uvPos, real2* uvVel, real4* cartPosVel )
{
   uint idx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
   if (idx>=numParticles)
      return;
      
   unsigned int objNum = (unsigned int)uvPos[idx].w;
   int surfaceType = allObjParams[objNum].obj_type;

   real u[2];
   real ud[2];
   real fp[3];
   real metric[4];

   u[0] = uvPos[idx].x;
   u[1] = uvPos[idx].y;
   ud[0] = uvVel[idx].x;
   ud[1] = uvVel[idx].y;

   CSObjParams* objParams = &allObjParams[objNum];

   // calculate position of the p-particle
   switch (surfaceType)
   {
      case SURF_TYPE_PLANE: {
         plane_calc_f(u,fp,objParams);
         plane_calc_metric(u,metric,objParams);
         break;
      }
      case SURF_TYPE_SPHERE: {
         sphere_calc_f(u,fp,objParams);
         sphere_calc_metric(u,metric,objParams);
         break;
      }
      case SURF_TYPE_ELLIPSOID: {
         break;
      }
      case SURF_TYPE_FRUSTUM: {
         frustum_calc_f(u,fp,objParams);
         frustum_calc_metric(u,metric,objParams);
         break;
      }
      case SURF_TYPE_TORUS: {
         torus_calc_f(u,fp,objParams);
         torus_calc_metric(u,metric,objParams);
         break;
      }
      case SURF_TYPE_MOEBIUS: {
         moebius_calc_f(u,fp,objParams);
         moebius_calc_metric(u,metric,objParams);
         break;
      }  
      case SURF_TYPE_GRAPH: {
         graph_calc_f(u,fp,objParams);
         graph_calc_metric(u,metric,objParams);
         break;
      }
   }

   real velSqr = metric[0]*ud[0]*ud[0]
               + REAL_TWO*metric[1]*ud[0]*ud[1]
               + metric[3]*ud[1]*ud[1];

   cartPosVel[idx] = make_real4(fp[0],fp[1],fp[2],velSqr);
}


/** Do jerk 
 * \param uvVel : uv-velocities of particles.
 * \param factor : yerk factor
 */
__global__ void jerk( unsigned int numParticles, real2* uvVel, real factor )
{
   uint idx = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;
   if (idx>=numParticles)
      return;
      
   real vx = uvVel[idx].x;
   real vy = uvVel[idx].y;
   
   real vel = rSqrt(SQR(vx) + SQR(vy));
   real edvel = REAL_ONE/vel;
   vx = vx*edvel;
   vy = vy*edvel;
   
   uvVel[idx] = make_real2( vx*vel*factor, vy*vel*factor );    
}


/** Round a / b to nearest higher integer value.
 * \param a
 * \param b
 */
uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}


/** Compute grid and thread block size for a given number of elements.
 * \param n
 * \param blockSize
 * \param numBlocks
 * \param numThreads
 */
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
    numThreads = min(blockSize, n);
    numBlocks = iDivUp(n, numThreads);
}


/**  CUDA kernel for particle simulation.
 * \param numParticles : number of particles.
 * \param params : simulation parameters.
 * \param allObjParams : object parameters for all objects.
 * \param currPos
 * \param currVel
 * \param currCol
 * \param newPos
 * \param newVel
 * \param newCol
 * \param mass : particle mass.
 * \param charge : particle charge.
 */
extern "C" void launch_kernel( unsigned int numParticles, CSparams* params, CSObjParams* allObjParams,
                               real4* currPos, real2* currVel, float4* currCol,
                               real4* newPos,  real2* newVel, float4* newCol,
                               real* mass, real* charge )
{
   uint numThreads, numBlocks;
   computeGridSize(numParticles, DEF_BLOCK_SIZE, numBlocks, numThreads);
   curvedSurfaceStep<<<numBlocks,numThreads>>>(numParticles,params,allObjParams,currPos,currVel,currCol, newPos,newVel,newCol, mass, charge);
}



/**  CUDA kernel to determine 3d positions from uv parameters.
 * \param numParticles : number of particles.
 * \param allObjParams : object parameters for all objects.
 * \param uvPos : uv-positions of particles.
 * \param cartPos : cartesian coordinates of particles.
 */
extern "C" void launch_kernelPositions( unsigned int numParticles, CSObjParams* allObjParams,
                                        real4* uvPos, real4* cartPos )
{
   uint numThreads, numBlocks;
   computeGridSize(numParticles, DEF_BLOCK_SIZE, numBlocks, numThreads);
   calc_f_forall<<<numBlocks,numThreads>>>(numParticles,allObjParams,uvPos,cartPos);
}



/**  CUDA kernel to determine 3d positions and velocities from uv parameters.
 * \param numParticles : number of particles.
 * \param allObjParams : object parameters for all objects.
 * \param uvPos : uv-positions of particles.
 * \param uvVel : uv-velocities of particles.
 * \param cartPos : cartesian coordinates of particles + velocity squared.
 */
extern "C" void launch_kernelPosVel( unsigned int numParticles, CSObjParams* allObjParams,
                                     real4* uvPos, real2* uvVel, real4* cartPosVel )
{
   uint numThreads, numBlocks;
   computeGridSize(numParticles, DEF_BLOCK_SIZE, numBlocks, numThreads);
   calc_fdfsqr_forall<<<numBlocks,numThreads>>>(numParticles,allObjParams,uvPos,uvVel,cartPosVel);
}


/**  CUDA kernel to give a jerk to each particle in the current direction 
 *   of motion or in an arbitrary direction.
 * \param numParticles : number of particles.
 * \param uvVel : uv-velocities of particles.
 */
extern "C" void launch_kernelJerk( unsigned int numParticles,
                                   real2* uvVel, real factor )
{
   uint numThreads, numBlocks;
   computeGridSize(numParticles, DEF_BLOCK_SIZE, numBlocks, numThreads);
   jerk<<<numBlocks,numThreads>>>(numParticles,uvVel,factor);
}



/**  Calculate surface coordinates from uv parameters.
 * \param surfaceType : surface type.
 * \param params : object parameters.
 * \param y
 * \param f
 */
extern "C" void  e_calcSurface( int surfaceType, CSObjParams* params, real* y, real* f ) {
   calcSurface(surfaceType,params,y,f);
}



/** Calculate surface normal at uv.
 * \param surfaceType : surface type.
 * \param params : object parameters.
 * \param y
 * \param n
 */
extern "C" void  e_calcNormal( int surfaceType, CSObjParams* params, real* y, real* n ) {
   calcNormal(surfaceType,params,y,n);
}

#endif //CURVED_SURFACE_CODE
