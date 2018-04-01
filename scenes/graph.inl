// --------------------------------------------
//  graph.inl
// --------------------------------------------

real   t_step = (real)1e-3;
real   m_damp = (real)100.0;

rvec2  ur = rvec2( 0.0, 2.0 );
rvec2  vr = rvec2( 0.0, 2.0 );


/**  Initialize object.
 * 
 */
void init_Objects()
{
   if (!mObjectList.empty())
      mObjectList.clear();
   
   Object  o;
   o.type    = e_surface_graph;
   
   o.u_range = ur;
   o.v_range = vr;
   o.use_modulo = glm::bvec2(false,false);
         
   o.slice_u = 80;
   o.slice_v = 80;
   
   o.center = rvec3(-0.5,-0.5,0);

   double n = 1.0;
   double m = 0.5;
   double a = 0.5;
   o.params.push_back(n);
   o.params.push_back(m);
   o.params.push_back(a);
   mObjectList.push_back(o);

   mds = 0.03;
}


/**  Set particles
 * 
 */
void  set_Particles()
{
   // filename for saving particle data (in the standard directory 'output')
   outFileName = "graph.dat";

   // Number of particles...
   mNumParticles = 128; 
   
   // Generate particle list with 
   //   { object number, initial particle number, final particle number } 
   ParticleList  pOnGraph = { 0,  0,  mNumParticles };   
   mParticleList.push_back(pOnGraph);
      
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
         
         // particle color
         *(optr++) = 1.0f;
         *(optr++) = 1.0f;
         *(optr++) = 0.0f;
         *(optr++) = 1.0f;
      
         // mass and charge in terms of electron properties
         *(mptr++) = (real)1;
         *(cptr++) = (real)10;
      }
   }
}   

/** Set supplementary stuff
 */
void  set_Supplement()
{
   mCamera.setEyePos(3.105,0.266,2.363);
   mCamera.setPOI(0.635,0.547,-0.100);
   mCamera.setDistance(3.5);
   mCamera.setPhi(-6.489);
   mCamera.setTheta(44.736);
   glutReshapeWindow(820,600);
}
