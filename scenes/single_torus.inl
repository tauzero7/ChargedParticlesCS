// --------------------------------------------
//  single_torus.inl
// --------------------------------------------

real   t_step = (real)2e-4;
real   m_damp = (real)50.0;

rvec2  ur = rvec2( 0.0, 2.0*M_PI );
rvec2  vr = rvec2( 0.0, 2.0*M_PI );


/**  Initialize torus object.
 * 
 */
void init_Objects()
{
   if (!mObjectList.empty())
      mObjectList.clear();
   
   Object  o;
   o.type    = e_surface_torus;
   
   o.u_range = ur;
   o.v_range = vr;
   o.uv_mod  = rvec2(2.0*M_PI,2.0*M_PI);
   o.use_modulo = glm::bvec2(true,true);
               
   o.slice_u = 200;
   o.slice_v = 100;

#if 1
   double radiusOut = 2.0;
   double radiusIn  = 0.9;
#else   
   double radiusOut = 1.0;
   double radiusIn  = radiusOut/5.0;
#endif
   
   o.params.push_back(radiusOut);
   o.params.push_back(radiusIn);
   o.wireFrame = false;
   
   mObjectList.push_back(o);
   
   mds = 0.08;
}

/**  Set particles
 * 
 */
void  set_Particles()
{
   outFileName = "torusParticles.dat";
   
   mNumParticles = 1024;
   
   // Generate particle list with  object number, initial and final particle number   
   ParticleList  pOnTorus = { 0, 0, mNumParticles };
   mParticleList.push_back(pOnTorus);
   
   h_particlePosition = new real[mNumParticles*4];
   h_particleVelocity = new real[mNumParticles*2];
   h_particleColor    = new float[mNumParticles*4];
   
   h_mass     = new real[mNumParticles];
   h_charge   = new real[mNumParticles];
   
   real*  pptr = &h_particlePosition[0];
   real*  vptr = &h_particleVelocity[0];
   float* optr = &h_particleColor[0];
   
   real*  mptr = &h_mass[0];
   real*  cptr = &h_charge[0];
   
   real u,v;
   for(unsigned int p=0; p<mParticleList.size(); p++)
   {
      for(unsigned int i=mParticleList[p].num_i; i<mParticleList[p].num_f; i++)
      {
         u = (real)( getRandomValue() * (ur[1]-ur[0]) + ur[0] );
         v = (real)( getRandomValue() * (vr[1]-vr[0]) + vr[0] );
      
         *(pptr++) = u;
         *(pptr++) = v;
         *(pptr++) = (real)0;      
         *(pptr++) = (real)mParticleList[p].obj_num; 
      
         *(vptr++) = (real)0;
         *(vptr++) = (real)0;
        
         // color
         *(optr++) = 1.0f;
         *(optr++) = 1.0f;
         *(optr++) = 0.0f;
         *(optr++) = 1.0f;   // color alpha    
         
         // charge
         *(cptr++) = (real)1;
         
         // mass    
         *(mptr++) = (real)1;         
      }
      
   }
}   

/** Set supplementary stuff
 */
void  set_Supplement()
{
   mCamera.setEyePos(6.731,-0.202,3.872);
   mCamera.setPOI(0,0,-0.54);
   mCamera.setDistance(8.05);
   mCamera.setPhi(-1.719);
   mCamera.setTheta(33.232);
   glutReshapeWindow(900,600);
}
