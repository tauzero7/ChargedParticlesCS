// --------------------------------------------
//  two_spheres.inl
// --------------------------------------------

real   t_step = (real)1e-2;
real   m_damp = (real)100.0;

rvec2  ur = rvec2( 0.0, 2.0*M_PI );
rvec2  vr = rvec2( 0.01, M_PI-0.01 );


/**  Initialize torus object.
 * 
 */
void init_Objects()
{
   if (!mObjectList.empty())
      mObjectList.clear();
   
// --- Sphere 1 ---   
   Object  o1;
   o1.type   = e_surface_sphere;
   
   o1.u_range = ur;
   o1.v_range = vr;
   o1.uv_mod  = rvec2(2.0*M_PI,M_PI);
   o1.use_modulo = glm::bvec2(true,false);
         
   o1.slice_u = 200;
   o1.slice_v = 100;
   
   o1.center  = rvec3(0,-1.1,0);
        
   double r = 1.0;
   o1.params.push_back(r);
   o1.wireFrame = false;


// --- Sphere 2 ---   
   Object  o2;
   o2.type   = e_surface_sphere;
   
   o2.u_range = ur;
   o2.v_range = vr;
   o2.uv_mod  = rvec2(2.0*M_PI,M_PI);
   o2.use_modulo = glm::bvec2(true,false);
         
   o2.slice_u = 200;
   o2.slice_v = 100;
   
   o2.center  = rvec3(0,0.5,0);
        
   r = 0.5;
   o2.params.push_back(r);
   o2.wireFrame = false;   
   
   mObjectList.push_back(o1);
   mObjectList.push_back(o2);
}


/**  Set particles
 * 
 */
void  set_Particles()
{
   outFileName = "twoSpheresParticles.dat";
   
   mNumParticles = 512; 
   
   // Generate particle list with  object number, initial and final particle number   
   int objCount = 0;
   ParticleList  pOnSphere1 = { (objCount++),   0,   mNumParticles/2 };
   ParticleList  pOnSphere2 = { (objCount++),  mNumParticles/2, mNumParticles };   
   mParticleList.push_back(pOnSphere1);
   mParticleList.push_back(pOnSphere2);
      
  // Allocate host memory for particles
   allocHostMemForParticles();
   
   real*  pptr = &h_particlePosition[0];
   real*  vptr = &h_particleVelocity[0];
   float* optr = &h_particleColor[0];   
   real*  mptr = &h_mass[0];
   real*  cptr = &h_charge[0];
   
   real u,v;
   real q = (real)1;

   //   srand(time(NULL));
   for(unsigned int p=0; p<mParticleList.size(); p++)
   {
      for(unsigned int i=mParticleList[p].num_i; i<mParticleList[p].num_f; i++)
      {
         u = (real)( getRandomValue() * (ur[1]-ur[0]) + ur[0] );
         v = (real)( getRandomValue() * (vr[1]-vr[0]) + vr[0] );
         //v = (real)( getRandomValue() * (M_PI-0.6) + 0.3 );
      
         *(pptr++) = u;
         *(pptr++) = v;
         *(pptr++) = (real)0;      
         *(pptr++) = (real)mParticleList[p].obj_num; 
      
         *(vptr++) = (real)0;
         *(vptr++) = (real)0;
         
         if (p==0) {
            *(optr++) = 1.0f;
            *(optr++) = 0.3f;
            *(optr++) = 0.3f;
            
            *(cptr++) = q;
         }
         else {
            *(optr++) = 0.3f;
            *(optr++) = 0.3f;
            *(optr++) = 1.0f;
            
            *(cptr++) = -q;
         }
         *(optr++) = 1.0f;
      
         *(mptr++) = (real)1;      
      }
   }
}   

/** Set supplementary stuff
 */
void  set_Supplement()
{
   mCamera.setEyePos(2.537,2.038,1.808);
   mCamera.setPOI(-0.017,-0.438,0.090);
   mCamera.setDistance(3.95);
   mCamera.setPhi(44.118);
   mCamera.setTheta(25.783);
   glutReshapeWindow(800,600);
}
