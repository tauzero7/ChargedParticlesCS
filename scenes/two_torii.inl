// --------------------------------------------
//  two_torii.inl
// --------------------------------------------

real   t_step = (real)5e-4;
real   m_damp = (real)2e2;

rvec2  ur = rvec2( 0.0, 2.0*M_PI );
rvec2  vr = rvec2( 0.0, 2.0*M_PI );


/**  Initialize torus object.
 * 
 */
void init_Objects()
{
   if (!mObjectList.empty())
      mObjectList.clear();

   double radiusOut;
   double radiusIn;
   
// --- Torus 1 ---
   Object  o1;
   o1.type   = e_surface_torus;
   
   o1.u_range = ur;
   o1.v_range = vr;
   o1.uv_mod  = rvec2(2.0*M_PI,2.0*M_PI);
   o1.use_modulo = glm::bvec2(true,true);
   
   o1.slice_u = 200;
   o1.slice_v = 100;
   
   o1.center  = rvec3(0,0,0);				
   o1.e1      = rvec3(1,0,0);
   o1.e2      = rvec3(0,1,0);
   o1.e3      = rvec3(0,0,1);
      
   radiusOut = 2.0;
   radiusIn  = 0.9;
   o1.params.push_back(radiusOut);
   o1.params.push_back(radiusIn);
   
   
// --- Torus 2 ---   
   Object  o2;
   o2.type    = e_surface_torus;
   
   o2.u_range = ur;
   o2.v_range = vr;
   o2.uv_mod  = rvec2(2.0*M_PI,2.0*M_PI);
   o2.use_modulo = glm::bvec2(true,true);
   
   o2.slice_u = 200;
   o2.slice_v = 100;
   
   o2.center  = rvec3(0,2,0);
   o2.e1      = rvec3(0,0,-1);
   o2.e2      = rvec3(0,1,0);
   o2.e3      = rvec3(1,0,0);
      
   radiusOut = 2.0;
   radiusIn  = 0.9;
   o2.params.push_back(radiusOut);
   o2.params.push_back(radiusIn);
   
   mObjectList.push_back(o1);
   mObjectList.push_back(o2);

   mds = 0.08;
}

/**  Set particles
 * 
 */
void  set_Particles()
{
   outFileName = "twoToriiParticles.dat";
   
   int numpart1  = 1024; 
   int numpart2  = 1024; 
   mNumParticles = numpart1 + numpart2;
   
   // Generate particle list with  object number, initial and final particle number
   int objCount = 0;
   ParticleList  pOnTorus1 = { (objCount++),   0, numpart1 };
   ParticleList  pOnTorus2 = { (objCount++), numpart1, mNumParticles };   
   mParticleList.push_back(pOnTorus1);
   mParticleList.push_back(pOnTorus2);
   
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
         
         //if (p==0 || p==1) {
         if (p==0) {
            // color
            *(optr++) = 1.0f;
            *(optr++) = 1.0f;
            *(optr++) = 0.0f;
            
            // charge
            *(cptr++) = (real)1;
         }
         else {
            // color
            *(optr++) = 0.5f;
            *(optr++) = 0.8f;
            *(optr++) = 1.0f;
            
            // charge
            *(cptr++) = -(real)1;
         }
         *(optr++) = 1.0f;   // color alpha
      
         *(mptr++) = (real)1;         
      }      
   }
}   

/** Set supplementary stuff
 */
void  set_Supplement()
{
   mCamera.setPOI(-0.222,0.740,0.090);
   mCamera.setDistance(11.2);
   //mCamera.setDistance(15.2);
   mCamera.setPhi(-24.637);
   mCamera.setTheta(33.232);
   glutReshapeWindow(760,600);
}
