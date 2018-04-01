// ---------------------------------------------------------------------
//   Frustum given in coordinates
//
//     u^1 = phi  and  u^2 = z
// ---------------------------------------------------------------------

/** Calculate vector function f(u1,u2)
 * \param u
 * \param f
 * \param o
 */
__host__ __device__ void  frustum_calc_f( real* u, real* f, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   real h  = o->value[2];
   
   real phi = u[0];
   real z   = u[1];
   
   real rho = r1 - (r1-r2)/h*z;
   
   real x1 = rho*cos(phi);
   real x2 = rho*sin(phi);
   real x3 = z;
   
   f[0] = x1*o->e1[0] + x2*o->e2[0] + x3*o->e3[0] + o->center[0];
   f[1] = x1*o->e1[1] + x2*o->e2[1] + x3*o->e3[1] + o->center[1];
   f[2] = x1*o->e1[2] + x2*o->e2[2] + x3*o->e3[2] + o->center[2];
}

/** Calculate surface normal n(u1,u2).
 * \param u
 * \param n
 * \param o
 */
__host__ __device__ void  frustum_calc_normal( real* u, real* n, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   real h  = o->value[2];
   
   real phi = u[0]; 
   real rhos = -(r1-r2)/h;
      
   real w = (real)1/sqrt(1+rhos*rhos);
   real nx = w*cos(phi);
   real ny = w*sin(phi);
   real nz = -rhos*w;
      
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
__device__ void  frustum_calc_df( real* u, real* dfdu1, real* dfdu2, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   real h  = o->value[2];
   
   real phi = u[0];
   real z   = u[1];
   
   real rho = r1 - (r1-r2)/h*z;
   real rhos = -(r1-r2)/h;
   
   // df/du1
   real dx1 = -rho*sin(phi);
   real dy1 = rho*cos(phi);
   real dz1 = (real)0;
   
   dfdu1[0] = dx1*o->e1[0] + dy1*o->e2[0] + dz1*o->e3[0];
   dfdu1[1] = dx1*o->e1[1] + dy1*o->e2[1] + dz1*o->e3[1];
   dfdu1[2] = dx1*o->e1[2] + dy1*o->e2[2] + dz1*o->e3[2];
  
   // df/du2
   real dx2 = rhos*cos(phi);
   real dy2 = rhos*sin(phi);
   real dz2 = (real)1;
   
   dfdu2[0] = dx2*o->e1[0] + dy2*o->e2[0] + dz2*o->e3[0];
   dfdu2[1] = dx2*o->e1[1] + dy2*o->e2[1] + dz2*o->e3[1]; 
   dfdu2[2] = dx2*o->e1[2] + dy2*o->e2[2] + dz2*o->e3[2];
}

/** Calculate metric coefficients.
 * \param u
 * \param metric
 * \param o
 */
__device__ void  frustum_calc_metric( real* u, real* metric, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   real h  = o->value[2];
      
   real z   = u[1];
   
   real rho = r1 - (r1-r2)/h*z;
   real rhos = -(r1-r2)/h;
   
   metric[0] = rho*rho;
   metric[1] = metric[2] = (real)0;
   metric[3] = rhos*rhos + (real)1;
}

/** Calculate inverse metric coefficients.
 * \param u
 * \param inmetric
 * \param o
 */
__device__ void  frustum_calc_invMetric( real* u, real* invmetric, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   real h  = o->value[2];
      
   real z   = u[1];
   
   real rho = r1 - (r1-r2)/h*z;
   real rhos = -(r1-r2)/h;
   
   invmetric[0] = 1.0/(rho*rho);
   invmetric[1] = invmetric[2] = (real)0;
   invmetric[3] = 1.0/(rhos*rhos+1);
}

/** Calculate christoffels of the second kind.
 * \param u
 * \param christoffel
 * \param o
 */
__device__ void  frustum_calc_chris( real* u, real* christoffel, CSObjParams* o )
{
   real r1 = o->value[0];
   real r2 = o->value[1];
   real h  = o->value[2];

   real z = u[1];

   real r1r2s = (r1-r2)*(r1-r2);
   real g112 = -(r1r2s + h*r1*(r2-r1))/(r1r2s+h*h);
   real g121 = (r2-r1)/((r2-r1)*z+h*r1);
   
   christoffel[0] = REAL_ZERO;                          // Gamma_00^0
   christoffel[1] = g121;                               // Gamma_01^0
   christoffel[2] = g121;                               // Gamma_10^0
   christoffel[3] = REAL_ZERO;                          // Gamma_11^0
  
   christoffel[4] = g112;                               // Gamma_00^1
   christoffel[5] = REAL_ZERO;                          // Gamma_01^1
   christoffel[6] = REAL_ZERO;                          // Gamma_10^1
   christoffel[7] = REAL_ZERO;                          // Gamma_11^1
}
