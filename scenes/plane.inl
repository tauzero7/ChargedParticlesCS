// --------------------------------------------
//  graph.inl
// --------------------------------------------

real   t_step = (real)5e-5;
real   m_damp = (real)200.0;

rvec2  ur = rvec2( -0.5, 0.5 );
rvec2  vr = rvec2( -0.5, 0.5 );


/**  Initialize object.
 * 
 */
void init_Objects()
{
   if (!mObjectList.empty())
      mObjectList.clear();
   
   Object  o1;
   o1.type    = e_surface_plane;   
   o1.u_range = ur;
   o1.v_range = vr;
   o1.center = rvec3(0,0,0);
   o1.e1     = rvec3(1,0,0);
   o1.e2     = rvec3(0,1,0);
   o1.e3     = rvec3(0,0,1);
      
   mObjectList.push_back(o1);  

   mds = 0.02f;
}


/**  Set particles
 * 
 */
void  set_Particles()
{
   // filename for saving particle data (in the standard directory 'output')
   outFileName = "graph.dat";

   // Number of particles...
   mNumParticles = 80; 
   
   // Generate particle list with 
   //   { object number, initial particle number, final particle number } 
   ParticleList  pOnPlane;
   pOnPlane.obj_num = 0;
   pOnPlane.num_i = 0;
   pOnPlane.num_f = mNumParticles;
   mParticleList.push_back(pOnPlane);
   
   
      
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
         
         
            *(optr++) = 1.0f;
            *(optr++) = 1.0f;
            *(optr++) = 0.0f;      
            *(cptr++) = (real)1;
         
         
         *(optr++) = 1.0f;
         *(mptr++) = (real)1;
      }
   }
}   

/** Set supplementary stuff
 */
void  set_Supplement()
{
   mCamera.setPhi(0);
   mCamera.setTheta(90);
   mCamera.setDistance(1.85);
}
