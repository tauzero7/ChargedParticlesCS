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
__host__ __device__ void  sphere_calc_f( real* u, real* f, CSObjParams* o )
{  
   real r = o->value[0];
   
   real phi   = u[0];
   real theta = u[1];
   
   real x = r*rSin(theta)*rCos(phi);
   real y = r*rSin(theta)*rSin(phi);
   real z = r*rCos(theta);  
   
   f[0] = x*o->e1[0] + y*o->e2[0] + z*o->e3[0] + o->center[0];
   f[1] = x*o->e1[1] + y*o->e2[1] + z*o->e3[1] + o->center[1];
   f[2] = x*o->e1[2] + y*o->e2[2] + z*o->e3[2] + o->center[2];
}

/** Calculate surface normal n(u1,u2).
 * \param u
 * \param n
 * \param o
 */
__host__ __device__ void  sphere_calc_normal( real* u, real* n, CSObjParams* o )
{
   real phi   = u[0];
   real theta = u[1];
   
   real nx = rSin(theta)*rCos(phi);
   real ny = rSin(theta)*rSin(phi);
   real nz = rCos(theta);
   
   n[0] = nx*o->e1[0] + ny*o->e2[0] + nz*o->e3[0];
   n[1] = nx*o->e1[1] + ny*o->e2[1] + nz*o->e3[1];
   n[2] = nx*o->e1[2] + ny*o->e2[2] + nz*o->e3[2];
}

/** Calculate derivative of vector functions
 *    df/du1(u1,u2) and df/du2(u1,u2)
 * \param u
 * \param dfdu1
 * \param dfdu2
 * \param o 
 */
__device__ void  sphere_calc_df( real* u, real* dfdu1, real* dfdu2, CSObjParams* o )
{
   real r = o->value[0];
   
   real phi   = u[0];
   real theta = u[1];
   real sintheta = rSin(theta);
   real costheta = rCos(theta);
   real sinphi   = rSin(phi);
   real cosphi   = rCos(phi);
   
   // df/dphi
   dfdu1[0] = -r*sintheta*sinphi; 
   dfdu1[1] =  r*sintheta*cosphi;
   dfdu1[2] =  REAL_ZERO;
   
   // df/dtheta
   dfdu2[0] = r*costheta*cosphi;
   dfdu2[1] = r*costheta*sinphi; 
   dfdu2[2] = -r*sintheta;
}

/** Calculate metric coefficients.
 * \param u
 * \param metric
 * \param o
 */
__device__ void  sphere_calc_metric( real* u, real* metric, CSObjParams* o )
{
   real r = o->value[0];
   
   real theta = u[1];
   real sintheta = rSin(theta);
   
   metric[0] = r*r*sintheta*sintheta;
   metric[1] = metric[2] = REAL_ZERO;
   metric[3] = r*r;
}

/** Calculate inverse metric coefficients.
 * \param u
 * \param inmetric
 * \param o
 */
__device__ void  sphere_calc_invMetric( real* u, real* invmetric, CSObjParams* o )
{   
   real r = o->value[0];
      
   real theta = u[1];
   real sintheta = rSin(theta);
      
   invmetric[0] = REAL_ONE/(r*r*sintheta*sintheta);
   invmetric[1] = invmetric[2] = REAL_ZERO;
   invmetric[3] = REAL_ONE/(r*r);
}

/** Calculate christoffels of the second kind.
 * \param u
 * \param christoffel
 * \param o
 */
__device__ void  sphere_calc_chris( real* u, real* christoffel, CSObjParams* o )
{
   real theta = u[1];

   real sintheta = rSin(theta);
   real costheta = rCos(theta);
   
   christoffel[0] = REAL_ZERO;                  // Gamma_00^0  
   christoffel[1] = costheta/sintheta;          // Gamma_01^0
   christoffel[2] = christoffel[1];             // Gamma_10^0
   christoffel[3] = REAL_ZERO;                  // Gamma_11^0
  
   christoffel[4] = -sintheta*costheta;         // Gamma_00^1
   christoffel[5] = REAL_ZERO;                  // Gamma_01^1
   christoffel[6] = REAL_ZERO;                  // Gamma_10^1
   christoffel[7] = REAL_ZERO;                  // Gamma_11^1
}
