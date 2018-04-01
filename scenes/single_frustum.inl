// --------------------------------------------
//  single_frustum.inl
// --------------------------------------------

double frustum_r1 = 2.0;
double frustum_r2 = 2.0;
double frustum_h  = 2.7;

real   t_step = (real)1e-2;
real   m_damp = (real)5.0;

rvec2  ur = rvec2( 0.0, 2.0*M_PI );
rvec2  vr = rvec2( 0.0, frustum_h );


/**  Initialize torus object.
 * 
 */
void init_Objects()
{
   if (!mObjectList.empty())
      mObjectList.clear();
 
   Object  o;
   o.type = e_surface_frustum;
   
   o.slice_u = 200;
   o.slice_v = 10;

   o.u_range = ur;
   o.v_range = vr;
   o.uv_mod  = rvec2(2.0*M_PI,1.0);
   o.use_modulo = glm::bvec2(true,false);       
      
   o.center = rvec3(0,0,-frustum_h*0.5);

   o.params.push_back(frustum_r1);
   o.params.push_back(frustum_r2);
   o.params.push_back(frustum_h);

   mObjectList.push_back(o);
      
   mds = 0.05f;
}


/**  Set particles
 * 
 */
void  set_Particles()
{
   mNumParticles = 32; 
   
   // Generate particle list with  object number, initial and final particle number   
   ParticleList  pOnFrustum = { 0,  0,  mNumParticles };   
   mParticleList.push_back(pOnFrustum);
      
   // Allocate host memory for particles
   allocHostMemForParticles();
   
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
         
         *(optr++) = 1.0f;
         *(optr++) = 1.0f;
         *(optr++) = 0.0f;
         *(optr++) = 1.0f;
      
         *(mptr++) = (real)1;
         *(cptr++) = (real)1;
      }
   }
}   

/** Set supplementary stuff
 */
void  set_Supplement()
{
   mCamera.setEyePos(6.496,-2.670,2.273);
   mCamera.setPOI(0.0,0.0,-1.12);
   mCamera.setDistance(7.8);
   mCamera.setPhi(-22.345);
   mCamera.setTheta(25.783);
}
