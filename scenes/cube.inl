// --------------------------------------------
//  graph.inl
// --------------------------------------------

real   t_step = (real)1e-2;
real   m_damp = (real)10.0;

rvec2  ur = rvec2( -0.5, 0.5 );
rvec2  vr = rvec2( -0.5, 0.5 );


/**  Initialize object.
 * 
 */
void init_Objects()
{
   if (!mObjectList.empty())
      mObjectList.clear();
   
   real offset = 0.05;
   
   Object  o1;
   o1.type    = e_surface_plane;   
   o1.u_range = ur;
   o1.v_range = vr;
   o1.center = rvec3(0,0,0.5+offset);
   o1.e1     = rvec3(1,0,0);
   o1.e2     = rvec3(0,1,0);
   o1.e3     = rvec3(0,0,1);
   
   Object  o2;
   o2.type    = e_surface_plane;   
   o2.u_range = ur;
   o2.v_range = vr;
   o2.center = rvec3(0,0,-0.5-offset);
   o2.e1     = rvec3(1,0,0);
   o2.e2     = rvec3(0,1,0);
   o2.e3     = rvec3(0,0,1);
   
   Object  o3;
   o3.type    = e_surface_plane;   
   o3.u_range = ur;
   o3.v_range = vr;
   o3.center = rvec3(0.5+offset,0,0);
   o3.e1     = rvec3(0,1,0);
   o3.e2     = rvec3(0,0,1);
   o3.e3     = rvec3(1,0,0);
   
   Object  o4;
   o4.type    = e_surface_plane;   
   o4.u_range = ur;
   o4.v_range = vr;
   o4.center = rvec3(-0.5-offset,0,0);
   o4.e1     = rvec3(0,1,0);
   o4.e2     = rvec3(0,0,1);
   o4.e3     = rvec3(1,0,0);
   
   Object  o5;
   o5.type    = e_surface_plane;   
   o5.u_range = ur;
   o5.v_range = vr;
   o5.center = rvec3(0,0.5+offset,0);
   o5.e1     = rvec3(-1,0,0);
   o5.e2     = rvec3(0,0,1);
   o5.e3     = rvec3(0,1,0);
   
   Object  o6;
   o6.type    = e_surface_plane;   
   o6.u_range = ur;
   o6.v_range = vr;
   o6.center = rvec3(0,-0.5-offset,0);
   o6.e1     = rvec3(-1,0,0);
   o6.e2     = rvec3(0,0,1);
   o6.e3     = rvec3(0,1,0);
   
   mObjectList.push_back(o1);   
   mObjectList.push_back(o2);
   mObjectList.push_back(o3);
   mObjectList.push_back(o4);
   mObjectList.push_back(o5);
   mObjectList.push_back(o6);

   mds = 0.05f;
}


/**  Set particles
 * 
 */
void  set_Particles()
{
   // filename for saving particle data (in the standard directory 'output')
   outFileName = "graph.dat";

   // Number of particles...
   mNumParticles = 240; 
   int nStep = mNumParticles / 6;
   
   // Generate particle list with 
   //   { object number, initial particle number, final particle number } 
   ParticleList  pOnCube[6];
   for(int i=0; i<6; i++) {
      pOnCube[i].obj_num = i;
      pOnCube[i].num_i = i*nStep;
      pOnCube[i].num_f = (i+1)*nStep;
      mParticleList.push_back(pOnCube[i]);
   }
   
      
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
         
         if (p % 2 == 0) {          
            *(optr++) = 1.0f;
            *(optr++) = 1.0f;
            *(optr++) = 0.0f;      
            *(cptr++) = (real)1;
         }
         else {          
            *(optr++) = 1.0f;
            *(optr++) = 0.0f;
            *(optr++) = 0.0f;      
            *(cptr++) = (real)1;
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
   // do nothing
}
