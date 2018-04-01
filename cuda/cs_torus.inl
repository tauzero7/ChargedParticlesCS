// ---------------------------------------------------------------------
//   Sphere given in standard spherical coordinates with
//
//     u^1 = theta  and  u^2 = phi
// ---------------------------------------------------------------------

/** Calculate vector function f(u1,u2)
 * \param u
 * \param f
 * \param o
 */
__host__ __device__ void  torus_calc_f( real* u, real* f, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   
   real theta = u[0];
   real phi   = u[1];
   
   real costheta = rCos(theta);
   real sintheta = rSin(theta);
   
   real x = (r1+r2*costheta)*rCos(phi);
   real y = (r1+r2*costheta)*rSin(phi);
   real z = r2*sintheta;
   
   f[0] = x*o->e1[0] + y*o->e2[0] + z*o->e3[0] + o->center[0];
   f[1] = x*o->e1[1] + y*o->e2[1] + z*o->e3[1] + o->center[1];
   f[2] = x*o->e1[2] + y*o->e2[2] + z*o->e3[2] + o->center[2];
}

/** Calculate surface normal n(u1,u2).
 * \param u
 * \param n
 * \param o
 */
__host__ __device__ void  torus_calc_normal( real* u, real* n, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   
   real theta = u[0];
   real phi   = u[1];
      
   real costheta = rCos(theta);
   real sintheta = rSin(theta);
   real cosphi   = rCos(phi);
   real sinphi   = rSin(phi);
   
   // -df/dtheta 
   real dx1 =  r2*sintheta*cosphi;
   real dy1 =  r2*sintheta*sinphi;
   real dz1 = -r2*costheta;
   
   // df/dphi
   real dx2 = -(r1+r2*costheta)*sinphi;
   real dy2 =  (r1+r2*costheta)*cosphi;
   real dz2 =  (real)0;
   
   // pointing inwards
   real nx = dy1*dz2 - dz1*dy2;
   real ny = dz1*dx2 - dx1*dz2;
   real nz = dx1*dy2 - dy1*dx2;
   
   real enl = REAL_ONE/rSqrt(nx*nx + ny*ny + nz*nz);
   nx *= enl;
   ny *= enl;
   nz *= enl;   
      
   n[0] = nx*o->e1[0] + ny*o->e2[0] + nz*o->e3[0];
   n[1] = nx*o->e1[1] + ny*o->e2[1] + nz*o->e3[1];
   n[2] = nx*o->e1[2] + ny*o->e2[2] + nz*o->e3[2];   
}

/** Calculate derivative of vector functions.
 *    df/du1(u1,u2) and df/du2(u1,u2)
 * \param u
 * \param dfdu1
 * \param dfdu2
 * \param o 
 */
__device__ void  torus_calc_df( real* u, real* dfdu1, real* dfdu2, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   
   real theta = u[0];
   real phi   = u[1];
      
   real costheta = rCos(theta);
   real sintheta = rSin(theta);
   real cosphi   = rCos(phi);
   real sinphi   = rSin(phi);
   
   // df/du1
   real dx1 = -r2*sintheta*cosphi;
   real dy1 = -r2*sintheta*sinphi;
   real dz1 =  r2*costheta;
   
   dfdu1[0] = dx1*o->e1[0] + dy1*o->e2[0] + dz1*o->e3[0];
   dfdu1[1] = dx1*o->e1[1] + dy1*o->e2[1] + dz1*o->e3[1];
   dfdu1[2] = dx1*o->e1[2] + dy1*o->e2[2] + dz1*o->e3[2];
  
   // df/du2
   real dx2 = -(r1+r2*costheta)*sinphi;
   real dy2 =  (r1+r2*costheta)*cosphi;
   real dz2 =  (real)0;
   
   dfdu2[0] = dx2*o->e1[0] + dy2*o->e2[0] + dz2*o->e3[0];
   dfdu2[1] = dx2*o->e1[1] + dy2*o->e2[1] + dz2*o->e3[1]; 
   dfdu2[2] = dx2*o->e1[2] + dy2*o->e2[2] + dz2*o->e3[2];
}

/** Calculate metric coefficients.
 * \param u
 * \param metric
 * \param o
 */
__device__ void  torus_calc_metric( real* u, real* metric, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   
   real theta = u[0];      
   real costheta = rCos(theta);
   
   metric[0] = r2*r2;
   metric[1] = metric[2] = 0.0;
   metric[3] = (r1+r2*costheta)*(r1+r2*costheta);
}

/** Calculate inverse metric coefficients.
 * \param u
 * \param inmetric
 * \param o
 */
__device__ void  torus_calc_invMetric( real* u, real* invmetric, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   
   real theta = u[0];
   real costheta = rCos(theta);
   
   invmetric[0] = 1.0/(r1*r1);
   invmetric[1] = invmetric[2] = 0.0;
   invmetric[3] = 1.0/((r1+r2*costheta)*(r1+r2*costheta));
}

/** Calculate christoffels of the second kind.
 * \param u
 * \param christoffel
 * \param o
 */
__device__ void  torus_calc_chris( real* u, real* christoffel, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   
   real theta = u[0];
   real costheta = rCos(theta);
   real sintheta = rSin(theta);
   
   christoffel[0] = 0.0;                                // Gamma_00^0
   christoffel[1] = 0.0;                                // Gamma_01^0
   christoffel[2] = 0.0;                                // Gamma_10^0
   christoffel[3] = (r1+r2*costheta)*sintheta/r2;       // Gamma_11^0
  
   christoffel[4] = 0.0;                                // Gamma_00^1
   christoffel[5] = -r2*sintheta/(r1+r2*costheta);      // Gamma_01^1
   christoffel[6] = christoffel[5];                     // Gamma_10^1
   christoffel[7] = 0.0;                                // Gamma_11^1
}
