// --------------------------------------------
//  single_sphere.inl						 
// --------------------------------------------
                                               
real   t_step = (real)2e-5;
real   m_damp = (real)500.0;

rvec2  ur = rvec2( 0.0, 2.0*M_PI );
rvec2  vr = rvec2( 0.01, M_PI-0.01 );


/**  Initialize object.
 * 
 */
void init_Objects()
{
   if (!mObjectList.empty())
      mObjectList.clear();
   
   Object  o;
   o.type    = e_surface_sphere;
   
   o.u_range = ur;
   o.v_range = vr;
   o.uv_mod  = rvec2(2.0*M_PI,M_PI);
   o.use_modulo = glm::bvec2(true,false);
         
   o.slice_u = 200;
   o.slice_v = 100;
   //o.slice_u = 50;
   //o.slice_v = 25;

   double r = 1.0;
   o.params.push_back(r);                                
   mObjectList.push_back(o);
   
   mds = 0.05f;
}


/**  Set particles
 * 
 */
void  set_Particles()
{
   // filename for saving particle data (in the standard directory 'output')
   outFileName = "singleSphere.dat";

   // Number of particles...
   mNumParticles = 128; 
   
   // Generate particle list with 
   //   { object number, initial particle number, final particle number } 
   ParticleList  pOnSphere = { 0,  0,  mNumParticles };   
   mParticleList.push_back(pOnSphere);
      
   // Allocate host memory for particles...
   allocHostMemForParticles();
   
   // Pointers for direct memory access...
   real*  pptr = &h_particlePosition[0];
   real*  vptr = &h_particleVelocity[0];
   float* optr = &h_particleColor[0];   
   real*  mptr = &h_mass[0];
   real*  cptr = &h_charge[0];
   
   real u,v;
   // Loop over all particle lists...
   for(unsigned int p=0; p<mParticleList.size(); p++)
   {
      // Loop over all particles in list 'p'...
      for(unsigned int i=mParticleList[p].num_i; i<mParticleList[p].num_f; i++)
      {
         u = (real)( getRandomValue() * (ur[1]-ur[0]) + ur[0] );
         v = (real)( getRandomValue() * (vr[1]-vr[0]) + vr[0] );
      
         // uv particle position, fixation, object number
         *(pptr++) = u;
         *(pptr++) = v;
         *(pptr++) = (real)0;      
         *(pptr++) = (real)mParticleList[p].obj_num; 
      
         // uv particle velocity
         *(vptr++) = (real)0;
         *(vptr++) = (real)0;
         
         
         if (i<mParticleList[p].num_f/2) {
            *(cptr++) = (real)10;
            *(optr++) = 1.0f;
            *(optr++) = 1.0f;
            *(optr++) = 0.0f;
            *(optr++) = 1.0f;   
         }
         else {
            *(cptr++) = (real)1;
            *(optr++) = 1.0f;
            *(optr++) = 0.0f;
            *(optr++) = 0.0f;
            *(optr++) = 1.0f;
         }
               
         // mass and charge in terms of electron properties
         *(mptr++) = (real)1;         
      }
   }

/*
   // Fix particle '0' at specific position...
   int num = 0;
   h_particlePosition[4*num+0] = (real)0.0;
   h_particlePosition[4*num+1] = (real)0.001;
   h_particlePosition[4*num+2] = (real)1;
   h_particleColor[4*num+0] = 1.0f;
   h_particleColor[4*num+1] = 0.0f;
   h_particleColor[4*num+2] = 0.0f;
   h_charge[num] = 50.0;
   
   // Fix particle '1' at specific position...
   num = 1;
   h_particlePosition[4*num+0] = (real)0.0;
   h_particlePosition[4*num+1] = (real)(M_PI-0.001);
   h_particlePosition[4*num+2] = (real)1;
   h_particleColor[4*num+0] = 0.0f;
   h_particleColor[4*num+1] = 0.0f;
   h_particleColor[4*num+2] = 1.0f;
   h_charge[num] = 50.0;
*/
}   

/** Set supplementary stuff
 */
void  set_Supplement()
{
   // do nothing
}
