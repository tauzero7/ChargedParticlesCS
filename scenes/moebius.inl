// --------------------------------------------
//  moebius.inl
// --------------------------------------------

real   t_step = (real)1e-2;
real   m_damp = (real)1.0;

rvec2  ur = rvec2( -0.8, 0.8 );
rvec2  vr = rvec2(  0.0, 1.999*PI );


/**  Initialize object.
 * 
 */
void init_Objects()
{
   if (!mObjectList.empty())
      mObjectList.clear();
   
   Object  o;
   o.type    = e_surface_moebius;
   
   o.u_range = ur;
   o.v_range = vr;
   o.uv_mod = rvec2(1.0,4.0*PI);
   o.use_modulo = glm::bvec2(false,false);
         
   o.slice_u = 20;
   o.slice_v = 80;
   
   o.center = rvec3(0,0,0);

   double n = 1.0;
   o.params.push_back(n);
   mObjectList.push_back(o);

   mds = 0.05f;
}


/**  Set particles
 * 
 */
void  set_Particles()
{
   // filename for saving particle data (in the standard directory 'output')
   outFileName = "moebius.dat";

   // Number of particles...
   mNumParticles = 256; 
   
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
         *(cptr++) = (real)1;
      }
   }
}   

/** Set supplementary stuff
 */
void  set_Supplement()
{
   // do nothing
}
