// ---------------------------------------------------------------------
//   Function graph given in coordinates u^1=u, u^2=v
//
//    \vec{f}(u,v) = ( u, v, g(u,v) )
//
//    Here,  g(u,v) = a^2 * sin(2*pi*n*u) * sin(2*pi*m*v)
// ---------------------------------------------------------------------

/** Calculate vector function f(u1,u2)
 * \param u
 * \param f
 * \param o
 */
__host__ __device__ void  graph_calc_f( real* u, real* f, CSObjParams* o )
{
   real n = o->value[0];
   real m = o->value[1];
   real a = o->value[2];
   
   real su = (real)(a*rSin(TWO_PI*n*u[0]));
   real sv = (real)(a*rSin(TWO_PI*m*u[1]));
   real g  = su*sv;
   
   f[0] = u[0] + o->center[0];
   f[1] = u[1] + o->center[1];
   f[2] = g + o->center[2];
}

/** Calculate surface normal n(u1,u2).
 * \param u
 * \param nr
 * \param o
 */
__host__ __device__ void  graph_calc_normal( real* u, real* nr, CSObjParams* o )
{
   real n = o->value[0];
   real m = o->value[1];
   real a = o->value[2];
   
   real su = (real)(rSin(TWO_PI*n*u[0]));
   real cu = (real)(rCos(TWO_PI*n*u[0]));
   real sv = (real)(rSin(TWO_PI*m*u[1]));
   real cv = (real)(rCos(TWO_PI*m*u[1]));
      
   real nx = -TWO_PI*n*a*a*cu*sv;
   real ny = -TWO_PI*m*a*a*su*cv;
   real nz = REAL_ONE;
   real ednorm = REAL_ONE/(SQR(nx)+SQR(ny)+SQR(nz));   
   nr[0] = nx*ednorm;
   nr[1] = ny*ednorm;
   nr[2] = nz*ednorm;
}

/** Calculate derivative of vector functions
 *    df/du1(u1,u2) and df/du2(u1,u2)
 * \param u
 * \param dfdu1
 * \param dfdu2
 * \param o 
 */
__device__ void  graph_calc_df( real* u, real* dfdu1, real* dfdu2, CSObjParams* o )
{
   real n = o->value[0];
   real m = o->value[1];
   real a = o->value[2];
   
   real su = (real)(rSin(TWO_PI*n*u[0]));
   real cu = (real)(rCos(TWO_PI*n*u[0]));
   real sv = (real)(rSin(TWO_PI*m*u[1]));
   real cv = (real)(rCos(TWO_PI*m*u[1]));   
   
   // df/du1  
   dfdu1[0] = REAL_ONE;
   dfdu1[1] = REAL_ZERO;
   dfdu1[2] = (real)(TWO_PI*n*a*a*cu*sv);
  
   // df/du2   
   dfdu2[0] = REAL_ZERO;
   dfdu2[1] = REAL_ONE;
   dfdu2[2] = (real)(TWO_PI*m*a*a*su*cv);
}

/** Calculate metric coefficients.
 * \param u
 * \param metric
 * \param o
 */
__device__ void  graph_calc_metric( real* u, real* metric, CSObjParams* o )
{
   real n = o->value[0];
   real m = o->value[1];
   real a = o->value[2];
   
   real su = (real)(rSin(TWO_PI*n*u[0]));
   real cu = (real)(rCos(TWO_PI*n*u[0]));
   real sv = (real)(rSin(TWO_PI*m*u[1]));
   real cv = (real)(rCos(TWO_PI*m*u[1]));   
   
   real a2 = a*a;
   real a4 = a2*a2;
   metric[0] = (real)(4*PI*PI*n*n*a4*cu*cu*sv*sv + 1);
   metric[1] = metric[2] = (real)(4*PI*PI*n*m*a4*su*cu*sv*cv);
   metric[3] = (real)(4*PI*PI*m*m*a4*su*su*cv*cv + 1);
}

/** Calculate inverse metric coefficients.
 * \param u
 * \param inmetric
 * \param o
 */
__device__ void  graph_calc_invMetric( real* u, real* invmetric, CSObjParams* o )
{
   real n = o->value[0];
   real m = o->value[1];
   real a = o->value[2];
   
   real su = (real)(rSin(TWO_PI*n*u[0]));
   real cu = (real)(rCos(TWO_PI*n*u[0]));
   real sv = (real)(rSin(TWO_PI*m*u[1]));
   real cv = (real)(rCos(TWO_PI*m*u[1]));  
   
   real a2 = a*a;
   real a4 = a2*a2;
   real HN = REAL_ONE/(4*PI*PI*n*n*a4*cu*cu*sv*sv + 4*PI*PI*m*m*a4*su*su*cv*cv + 1);
                      
   invmetric[0] = (4*PI*PI*m*m*a4*su*su*cv*cv+1)*HN;
   invmetric[1] = invmetric[2] = -((4*PI*PI*m*n*a4*cu*su*cv*sv)*HN);
   invmetric[3] = (4*PI*PI*n*n*a4*cu*cu*sv*sv+1)*HN;
}

/** Calculate christoffels of the second kind.
 * \param u
 * \param christoffel
 * \param o
 */
__device__ void  graph_calc_chris( real* u, real* christoffel, CSObjParams* o )
{
   real n = o->value[0];
   real m = o->value[1];
   real a = o->value[2];
   
   real su = (real)(rSin(TWO_PI*n*u[0]));
   real cu = (real)(rCos(TWO_PI*n*u[0]));
   real sv = (real)(rSin(TWO_PI*m*u[1]));
   real cv = (real)(rCos(TWO_PI*m*u[1]));  
   
   real a2 = a*a;
   real a4 = a2*a2;            
   real HN = REAL_ONE/(4*PI*PI*n*n*a4*cu*cu*sv*sv + 4*PI*PI*m*m*a4*su*su*cv*cv + 1);
   real PI3 = PI*PI*PI;            
                     
   // Gamma_00^0
   christoffel[0] = -(8*PI3*n*n*n*a4*cu*su*sv*sv)*HN; 
   
   // Gamma_01^0
   christoffel[1] = (8*PI3*m*n*n*a4*cu*cu*cv*sv)*HN;
   
   // Gamma_10^0
   christoffel[2] = christoffel[1];
   
   // Gamma_11^0
   christoffel[3] = -(8*PI3*m*m*n*a4*cu*su*sv*sv)*HN; 
  
   // Gamma_00^1
   christoffel[4] = -(8*PI3*m*n*n*a4*su*su*cv*sv)*HN;
   
   // Gamma_01^1
   christoffel[5] = (8*PI3*m*m*n*a4*cu*su*cv*cv)*HN; 
   
   // Gamma_10^1
   christoffel[6] = christoffel[5]; 
   
   // Gamma_11^1
   christoffel[7] = -(8*PI3*m*m*m*a4*su*su*cv*sv)*HN; 
}
