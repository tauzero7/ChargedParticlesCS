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
__host__ __device__ void  moebius_calc_f( real* u, real* f, CSObjParams* o )
{
   real n = o->value[0];
   
   real r   = u[0];
   real phi = u[1];
   
   real cosphi = rCos(phi);
   real sinphi = rSin(phi);
   real cp = rCos(0.5*n*phi);
   real sp = rSin(0.5*n*phi);
   
   real x1 = (1+0.5*r*cp)*cosphi;
   real x2 = (1+0.5*r*cp)*sinphi;
   real x3 = 0.5*r*sp;
   
   f[0] = x1*o->e1[0] + x2*o->e2[0] + x3*o->e3[0] + o->center[0];
   f[1] = x1*o->e1[1] + x2*o->e2[1] + x3*o->e3[1] + o->center[1];
   f[2] = x1*o->e1[2] + x2*o->e2[2] + x3*o->e3[2] + o->center[2];
}

/** Calculate surface normal n(u1,u2).
 * \param u
 * \param n
 * \param o
 */
__host__ __device__ void  moebius_calc_normal( real* u, real* nr, CSObjParams* o )
{
   real n = o->value[0];
   
   real r   = u[0];
   real phi = u[1];
   
   real cosphi = rCos(phi);
   real sinphi = rSin(phi);
   real cp = rCos(0.5*n*phi);
   real sp = rSin(0.5*n*phi);
      
   real nx = (n*sinphi*cp*cp*r)*0.125-(sp*(cosphi*((cp*r)*0.5+1)-(n*sinphi*sp*r)*0.25))*0.5;
   real ny = (sp*(-sinphi*((cp*r)*0.5+1)-(n*cosphi*sp*r)*0.25))*0.5-(n*cosphi*cp*cp*r)*0.125;
   real nz = (cosphi*cp*(cosphi*((cp*r)*0.5+1)-(n*sinphi*sp*r)*0.25))*0.5-(sinphi*cp*(-sinphi*((cp*r)*0.5+1)-(n*cosphi*sp*r)*0.25))*0.5;
      
   real nn = REAL_ONE/rSqrt(nx*nx+ny*ny+nz*nz);
   nx *= nn;   
   ny *= nn;
   nz *= nn;
      
   nr[0] = nx*o->e1[0] + ny*o->e2[0] + nz*o->e3[0];
   nr[1] = nx*o->e1[1] + ny*o->e2[1] + nz*o->e3[1];
   nr[2] = nx*o->e1[2] + ny*o->e2[2] + nz*o->e3[2];   
}

/** Calculate derivative of vector functions
 *    df/du1(u1,u2) and df/du2(u1,u2)
 * \param u
 * \param dfdu1
 * \param dfdu2
 * \param o 
 */
__device__ void  moebius_calc_df( real* u, real* dfdu1, real* dfdu2, CSObjParams* o )
{
   real n = o->value[0];
   
   real r   = u[0];
   real phi = u[1];
   
   real cosphi = rCos(phi);
   real sinphi = rSin(phi);
   real cp = rCos(0.5*n*phi);
   real sp = rSin(0.5*n*phi);
   
   // df/du1
   real dx1 = (cosphi*cp)*0.5;
   real dy1 = (sinphi*cp)*0.5;
   real dz1 = sp*0.5;
   
   dfdu1[0] = dx1*o->e1[0] + dy1*o->e2[0] + dz1*o->e3[0];
   dfdu1[1] = dx1*o->e1[1] + dy1*o->e2[1] + dz1*o->e3[1];
   dfdu1[2] = dx1*o->e1[2] + dy1*o->e2[2] + dz1*o->e3[2];
  
   // df/du2
   real dx2 = -sinphi*((cp*r)*0.5+1)-(n*cosphi*sp*r)*0.25;
   real dy2 = cosphi*((cp*r)*0.5+1)-(n*sinphi*sp*r)*0.25;
   real dz2 = (n*cp*r)*0.25;
   
   dfdu2[0] = dx2*o->e1[0] + dy2*o->e2[0] + dz2*o->e3[0];
   dfdu2[1] = dx2*o->e1[1] + dy2*o->e2[1] + dz2*o->e3[1]; 
   dfdu2[2] = dx2*o->e1[2] + dy2*o->e2[2] + dz2*o->e3[2];
}

/** Calculate metric coefficients.
 * \param u
 * \param metric
 * \param o
 */
__device__ void  moebius_calc_metric( real* u, real* metric, CSObjParams* o )
{
   real n = o->value[0];
   
   real r   = u[0];
   real phi = u[1];
   
   real cosphi = rCos(phi);
   real sinphi = rSin(phi);
   real cp = rCos(0.5*n*phi);
   real sp = rSin(0.5*n*phi);
   
   metric[0] = 0.25;
   metric[1] = metric[2] = REAL_ZERO;
   metric[3] = (cp*cp*r*r)*0.25+(n*n*r*r)*0.0625+cp*r+1;
}

/** Calculate inverse metric coefficients.
 * \param u
 * \param inmetric
 * \param o
 */
__device__ void  moebius_calc_invMetric( real* u, real* invmetric, CSObjParams* o )
{
   real n = o->value[0];
   
   real r   = u[0];
   real phi = u[1];
   
   real cosphi = rCos(phi);
   real sinphi = rSin(phi);
   real cp = rCos(0.5*n*phi);
   real sp = rSin(0.5*n*phi);
   
   invmetric[0] = (real)4;
   invmetric[1] = invmetric[2] = (real)0;
   invmetric[3] = (real)16/((4*cp*cp+n*n)*r*r+16*cp*r+16);
}

/** Calculate christoffels of the second kind.
 * \param u
 * \param christoffel
 * \param o
 */
__device__ void  moebius_calc_chris( real* u, real* christoffel, CSObjParams* o )
{
   real n = o->value[0];
   
   real r   = u[0];
   real phi = u[1];
   
   real cosphi = rCos(phi);
   real sinphi = rSin(phi);
   real cp = rCos(0.5*n*phi);
   real sp = rSin(0.5*n*phi);
   
   // Gamma_00^0
   christoffel[0] = REAL_ZERO;
   
   // Gamma_01^0
   christoffel[1] = REAL_ZERO;
   
   // Gamma_10^0
   christoffel[2] = christoffel[1];
   
   // Gamma_11^0
   christoffel[3] = -((4*cp*cp+n*n)*r+8*cp)*0.25;
  
   // Gamma_00^1
   christoffel[4] = REAL_ZERO;
   
   // Gamma_01^1   
   christoffel[5] = ((4*cp*cp+n*n)*r+8*cp)/((4*cp*cp+n*n)*r*r+16*cp*r+16);
   
   // Gamma_10^1
   christoffel[6] = christoffel[5];
   
   // Gamma_11^1
   christoffel[7] = -(2*n*cp*sp*r*r+4*n*sp*r)/((4*cp*cp+n*n)*r*r+16*cp*r+16);
}
