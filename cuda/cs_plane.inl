// ---------------------------------------------------------------------
//   Plane given in standard Cartesian coordinates with
//
//     u^1 = x  and  u^2 = y
// ---------------------------------------------------------------------

/** Calculate vector function f(u1,u2)
 * \param u
 * \param f
 * \param o
 */
__host__ __device__ void  plane_calc_f( real* u, real* f, CSObjParams* o )
{  
   real x = u[0];
   real y = u[1];
   
   f[0] = x*o->e1[0] + y*o->e2[0] + o->center[0];
   f[1] = x*o->e1[1] + y*o->e2[1] + o->center[1];
   f[2] = x*o->e1[2] + y*o->e2[2] + o->center[2];
}

/** Calculate surface normal n(u1,u2).
 * \param u
 * \param n
 * \param o
 */
__host__ __device__ void  plane_calc_normal( real* u, real* n, CSObjParams* o )
{
   n[0] = o->e3[0];
   n[1] = o->e3[1];
   n[2] = o->e3[2];
}

/** Calculate derivative of vector functions
 *    df/du1(u1,u2) and df/du2(u1,u2)
 * \param u
 * \param dfdu1
 * \param dfdu2
 * \param o 
 */
__device__ void  plane_calc_df( real* u, real* dfdu1, real* dfdu2, CSObjParams* o )
{   
   float dx1 = REAL_ONE;
   float dy1 = REAL_ZERO;
   float dz1 = REAL_ZERO;

   // df/dx
   dfdu1[0] = dx1*o->e1[0] + dy1*o->e2[0] + dz1*o->e3[0];
   dfdu1[1] = dx1*o->e1[1] + dy1*o->e2[1] + dz1*o->e3[1];
   dfdu1[2] = dx1*o->e1[2] + dy1*o->e2[2] + dz1*o->e3[2];
   
   float dx2 = REAL_ZERO;
   float dy2 = REAL_ONE;
   float dz2 = REAL_ZERO;
   
   // df/dtheta
   dfdu2[0] = dx2*o->e1[0] + dy2*o->e2[0] + dz2*o->e3[0];
   dfdu2[1] = dx2*o->e1[1] + dy2*o->e2[1] + dz2*o->e3[1]; 
   dfdu2[2] = dx2*o->e1[2] + dy2*o->e2[2] + dz2*o->e3[2];
}

/** Calculate metric coefficients.
 * \param u
 * \param metric
 * \param o
 */
__device__ void  plane_calc_metric( real* u, real* metric, CSObjParams* o )
{
   metric[0] = 1.0;
   metric[1] = metric[2] = 0.0;
   metric[3] = 1.0;
}

/** Calculate inverse metric coefficients.
 * \param u
 * \param inmetric
 * \param o
 */
__device__ void  plane_calc_invMetric( real* u, real* invmetric, CSObjParams* o )
{     
   invmetric[0] = 1.0;
   invmetric[1] = invmetric[2] = 0.0;
   invmetric[3] = 1.0;
}

/** Calculate christoffels of the second kind.
 * \param u
 * \param christoffel
 * \param o
 */
__device__ void  plane_calc_chris( real* u, real* christoffel, CSObjParams* o )
{
   christoffel[0] = 0.0;                        // Gamma_00^0  
   christoffel[1] = 0.0;                        // Gamma_01^0
   christoffel[2] = 0.0;                        // Gamma_10^0
   christoffel[3] = 0.0;                        // Gamma_11^0
  
   christoffel[4] = 0.0;                        // Gamma_00^1
   christoffel[5] = 0.0;                        // Gamma_01^1
   christoffel[6] = 0.0;                        // Gamma_10^1
   christoffel[7] = 0.0;                        // Gamma_11^1
}
