/*
   Copyright (c) 2012  Thomas Mueller

   This file is part of ChaPaCS.

   ChaPaCS is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ChaPaCS is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with ChaPaCS.  If not, see <http://www.gnu.org/licenses/>.
*/

/*  \file main.cpp

   \main Charged particle motion constrained to curved surfaces.
         The particles interact in 3D by means of the Coulomb force.

         Details to the CUDA library can be found at
         http://www.clear.rice.edu/comp422/resources/cuda/html/index.html

         To select a predefined scene, go to main.inl and adjust
         the entry 'USE_SCENE'. There, you can also change the
         background color.

         Depending on your system, you might have to adjust some
         texture sizes:

            FBO_TEXTURE_SIZE         512, 1024, 2048, ...
            
         Log output: t_curr,Ekin,Efield, [coc.x,coc.y,coc.z]

            current time, kinetic energy, field energy,
                        [ center of charge (x,y,z) ]
 */

#ifdef _WIN32
#include <Windows.h>
#endif

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cmath>

extern "C" {
#include <GL3/gl3w.h>
}

#include <cuda_gl_interop.h>

#include <defs.h>
#include <Camera.h>
#include <GLShader.h>
#include <RenderText.h>
#include <utils.h>

#include <GL/freeglut.h>
#include <cuda/header.cuh>

// You can set this value also in the Makefile
#ifndef FBO_TEXTURE_SIZE
#define FBO_TEXTURE_SIZE  512
#endif

// By default, potential rendering is not activated.
#ifndef FBO_POTI_TEXTURE_SIZE
#define FBO_POTI_TEXTURE_SIZE 512
#endif

// --------------------------------------------------------------------
//   Some prototype or externally defined functions.
//   See 'curvedSurfaceCode.cu' for their implementation.
// --------------------------------------------------------------------
extern "C" void launch_kernel ( unsigned int numParticles,
                                CSparams* params, CSObjParams* allObjParams,
                                real4* currPos, real2* currVel, float4* currCol,
                                real4* newPos,  real2* newVel,  float4* newCol,
                                real* mass, real* charge );

extern "C" void launch_kernelPositions ( unsigned int numParticles,
                                         CSObjParams* allObjParams,
                                         real4* uvPos, real4* cartPos );

extern "C" void launch_kernelPosVel( unsigned int numParticles,
                                     CSObjParams* allObjParams,
                                     real4* uvPos, real2* uvVel,  real4* cartPos );

extern "C" void launch_kernelJerk( unsigned int numParticles,
                                   real2* uvVel, real factor );

extern "C" void e_calcSurface ( int surfaceType, CSObjParams* params, real* y, real* f );
extern "C" void e_calcNormal  ( int surfaceType, CSObjParams* params, real* y, real* n );

void allocHostMemForParticles();


// -------------------------------------
//   global variables and definitions
// -------------------------------------
GLsizei  mWindowWidth  = 600; //800;
GLsizei  mWindowHeight = 600; //512;
bool     mPlay = false;

bool     mShowBox    = false;
bool     mShowWired  = false;
bool     mShowObjs   = true;
bool     mShowPoti   = false;
bool     mShowFLines = false;
bool     mShowHelp   = false;
int      mShowFBO    = -1;

float    mds = 0.03f;   //!< particle size

// initial particle visualization type
e_particleVisType  mParticleVis = e_particleVis_splats_only;

// Camera and mouse control
Camera   mCamera;
int      buttonPressed_ = -1;
int      xlast_ = -1;
int      ylast_ = -1;

// Current simulation time and step counter
real     t_curr = 0.0;
int      mCountSteps = 0;

// Object and particle lists
std::vector<Object>        mObjectList;
std::vector<ParticleList>  mParticleList;

// For text output
RenderText*  mRenderText = NULL;

// Vertex arrays and buffer objects for box and object drawing
GLuint  va_box;           //!< Vertex array handle for box
GLuint  vbo_box;          //!< Vertex buffer object handle for box
GLuint  va_object;        //!< Vertex array handle for all objects
GLuint  vbo_object;       //!< Vertex buffer object handle for all objects

// -------------------------------------
//   Standard shaders
// -------------------------------------
GLShader mShader;
const std::string  vShaderFileName = "shaders/curvedSurface.vert";
const std::string  fShaderFileName = "shaders/curvedSurface.frag";

GLShader mHelpShader;
const std::string  vHelpShaderFileName = "shaders/help.vert";
const std::string  fHelpShaderFileName = "shaders/help.frag";

const int maxShadingTypes = 4;     //!< Object shading defined in 'curvedSurface.fag'
int       mWhichShading   = 3;     //!< Current object shading.

// -------------------------------------
//   Framebuffer objects and shaders
// -------------------------------------
// ... particle mapping onto domain "U"
GLShader mFBOShader;
const std::string  vFBOShaderFileName = "shaders/particleMapping.vert";
const std::string  gFBOShaderFileName = "shaders/particleMapping.geom";
const std::string  fFBOShaderFileName = "shaders/particleMapping.frag";

// ... show fbo borders
GLShader mFBOTestShader;
const std::string  vFBOtestShaderFileName = "shaders/fboTest.vert";
const std::string  fFBOtestShaderFileName = "shaders/fboTest.frag";

// ... shaders for 'particle-textures'
GLShader mFBOShowShader;
const std::string  vFBOshowShaderFileName = "shaders/fboShow.vert";
const std::string  fFBOshowShaderFileName = "shaders/fboShow.frag";

GLuint* mFBOTexList = NULL;        //!< list of fbo texture handles
GLuint* mFBOList    = NULL;        //!< list of fbo handles

float mFBOoffsetX = 0.1f;          //!< x offset for fbo texture
float mFBOoffsetY = 0.1f;          //!< y offset for fbo texture

// ... shaders and fbo for potential mapping
GLShader mFBOPotentialShader;
const std::string vFBOpotShaderFileName = "shaders/fboPotential.vert";
const std::string fFBOpotShaderFileName = "shaders/fboPotential.frag";

GLuint* mFBOpotiTexList = NULL;
GLuint* mFBOpotiList    = NULL;

// ... fbo for image sequence output
GLuint  mFBOoutput      = 0;
GLuint  mFBOtexOutput   = 0;
GLuint  mFBOdepthOutput = 0;
bool    mRenderImageSeq = false;


// -------------------------------------
//   CUDA
// -------------------------------------
GLuint  va_particles[2];            //!< Vertex array handle for particle motion
GLuint  vbo_pos[2] = {0,0};         //!< Vertex buffer object for particle positions
GLuint  nbo_vel[2] = {0,0};         //!< Vertex buffer object for particle velocities
GLuint  cbo_col[2] = {0,0};         //!< Vertex buffer object for particle colors

struct cudaGraphicsResource *cuda_vbo_pos[2] = {NULL,NULL};
struct cudaGraphicsResource *cuda_nbo_vel[2] = {NULL,NULL};
struct cudaGraphicsResource *cuda_cbo_col[2] = {NULL,NULL};
int  curr_buf = 0;
int  new_buf  = 1;

unsigned int mNumParticles = 0;
real*        h_particlePosition = NULL;       //!< (u,v,isFixed,objNum)
real*        h_particleVelocity = NULL;       //!< (du,dv)
float*       h_particleColor    = NULL;       //!< (r,g,b,a)
real*        h_mass     = NULL;               //!< (mass)
real*        h_charge   = NULL;               //!< (charge)

CSparams    params;
CSparams    *d_params = NULL;
real        *d_mass   = NULL;
real        *d_charge = NULL;

CSObjParams *objParams    = NULL;
CSObjParams *d_obj_params = NULL;

// Standard output/input
std::string  outFileName = "particles.dat";   //!< standard output filename when pressing 'F','G'
std::string  inFileName  = std::string();     //!< if file is loaded, then 'reset' reloads this file.

// Log file for energy output...
std::string  logEfileName = "output/log_energy.dat";
FILE*        logEfilePtr  = NULL;
bool         mLogEnergy   = false;
unsigned int mLogSteps    = 1;

// Kinetic and field energy
real         mCurrEkin,mCurrEfield;

// Add initial velocity component to each particle
bool         mAddVelSet   = false;
real         mAddVelocity[2] = {REAL_ZERO,REAL_ZERO};

// -------------------------------------
//  data for calculation of field lines
// -------------------------------------
GLuint  va_fieldLines;
GLuint  vbo_fieldLines;
int*    h_numPointsPerFieldLine = NULL;
real*   h_fieldLines = NULL;

int mMaxNumPoints = 800;         //!< Maximum number of points a single field line consists of.
real           dl = (real)0.01;  //!< Step size for field line integration.


// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include "main.inl"
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/** Initialize OpenGL...
 *   and generate vertex arrays for objects and particles.
 */
void init_OpenGL()
{
   if (gl3wInit()) {
      fprintf(stderr,"Error: Failed to initialize gl3w.\n");
      exit(1);
   }
   glClearColor(bgColor[0],bgColor[1],bgColor[2],0.0f);
   glClearDepth(1.0);

   glEnable(GL_DEPTH_TEST);

   mRenderText = new RenderText("Font.ttf",13);

   // ------------------------------
   //  gen vertex array for object
   // ------------------------------
   glGenVertexArrays(1,&va_object);
   glBindVertexArray(va_object);
      glGenBuffers(1,&vbo_object);
      glBindBuffer(GL_ARRAY_BUFFER,vbo_object);
      glEnableVertexAttribArray(0);
      glBufferData(GL_ARRAY_BUFFER,sizeof(GLfloat)*4*4,quadVerts,GL_STATIC_DRAW);
      glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,NULL);
      glBindBuffer(GL_ARRAY_BUFFER,0);
   glBindVertexArray(0);

   // ------------------------------
   // vertex array for particles
   // ------------------------------
   glGenVertexArrays(2,va_particles);
}

/** Initialize framebuffer objects for mapping of particles onto surfaces.
 *    For each object we need a FBO with a texture.
 */
void init_FBOs()
{
   unsigned int num = (GLsizei)mObjectList.size();
   if (num==0)
      return;

   fprintf(stderr,"\nCreate %d FBOs of size %dx%d...\n",
           num,FBO_TEXTURE_SIZE,FBO_TEXTURE_SIZE);

   mFBOTexList = new GLuint[num];
   mFBOList    = new GLuint[num];

   glGenTextures(num,mFBOTexList);
   for(unsigned int i=0; i<num; i++) {
      glBindTexture(GL_TEXTURE_2D, mFBOTexList[i]);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      GLCE(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
                        FBO_TEXTURE_SIZE,FBO_TEXTURE_SIZE, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL));
   }

   glGenFramebuffers(num,mFBOList);
   for(unsigned int i=0; i<num; i++) {
      glBindFramebuffer( GL_FRAMEBUFFER, mFBOList[i] );
      glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mFBOTexList[i], 0 );

      GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
      switch (status)
      {
         case GL_FRAMEBUFFER_COMPLETE:
            fprintf(stderr,"\tFBO %d complete.\n",i);
            break;
         case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            fprintf(stderr,"\tFBO %d attachment incomplete.\n",i);
            break;
      }
   }
   glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}


/** Initialize framebuffer objects for mapping of particles onto surfaces.
 *    For each object we need a FBO with a texture.
 */
#ifdef WITH_POTENTIAL 
void init_FBOpotis()
{
   unsigned int num = (GLsizei)mObjectList.size();
   if (num==0)
      return;

   fprintf(stderr,"\nCreate %d FBOpoti(s) of size %dx%d ...\n",
         num,FBO_POTI_TEXTURE_SIZE,FBO_POTI_TEXTURE_SIZE);

   mFBOpotiTexList = new GLuint[num];
   mFBOpotiList    = new GLuint[num];

   glGenTextures(num,mFBOpotiTexList);
   for(unsigned int i=0; i<num; i++) {
      glBindTexture(GL_TEXTURE_2D, mFBOpotiTexList[i]);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      GLCE(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 
                        FBO_TEXTURE_SIZE,FBO_TEXTURE_SIZE, 0, GL_RGB, GL_FLOAT, NULL));
   }

   glGenFramebuffers(num,mFBOpotiList);
   for(unsigned int i=0; i<num; i++) {
      glBindFramebuffer( GL_FRAMEBUFFER, mFBOpotiList[i] );
      glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mFBOpotiTexList[i], 0 );

      GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
      switch (status)
      {
         case GL_FRAMEBUFFER_COMPLETE:
            fprintf(stderr,"\tFBOpoti %d complete.\n",i);
            break;
         case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
            fprintf(stderr,"\tFBOpoti %d attachment incomplete.\n",i);
            break;
      }
   }
   glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}
#endif

/** Initialize framebuffer objects for image output.
 */
void init_FBOoutput()
{
   fprintf(stderr,"Create FBO for output ... ");

   if (mFBOtexOutput!=0)
      glDeleteTextures(1,&mFBOtexOutput);

   glGenTextures(1,&mFBOtexOutput);
   glBindTexture(GL_TEXTURE_2D, mFBOtexOutput);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   GLCE(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 
                     mWindowWidth,mWindowHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL));

   if (mFBOoutput==0)  
      GLCE(glGenFramebuffers(1,&mFBOoutput));
   glBindFramebuffer( GL_FRAMEBUFFER, mFBOoutput );
      
   if (mFBOdepthOutput==0) 
      GLCE(glGenRenderbuffers(1,&mFBOdepthOutput));
   glBindRenderbuffer( GL_RENDERBUFFER, mFBOdepthOutput );
   
   GLCE(glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, mWindowWidth, mWindowHeight ));
   GLCE(glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, mFBOdepthOutput ));
   
   GLCE(glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, mFBOtexOutput, 0 ));

   GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
   switch (status)
   {
      default:
         fprintf(stderr,"ERROR.\n");
         break;
      case GL_FRAMEBUFFER_COMPLETE:
         fprintf(stderr,"complete.\n");
         break;
      case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
         fprintf(stderr,"ERROR: attachment incomplete.\n");
         break;
   }
   glBindFramebuffer( GL_FRAMEBUFFER, 0 );
}

/** Initialize CUDA device.
 * \param showProp : show properties of device.
 */
void init_CUDA( bool showProp = false )
{
   int deviceCount;
   CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));
   if (deviceCount == 0) {
      fprintf(stderr,"\nCUDA error: no devices supporting CUDA.\n");
      exit(-1);
   }
   else {
      fprintf(stderr,"\n# CUDA devices found: %d\n",deviceCount);
   }

   int dev;
   struct cudaDeviceProp  deviceProp;
   CUDA_SAFE_CALL(cudaGetDevice(&dev));
   CUDA_SAFE_CALL(cudaGetDeviceProperties(&deviceProp,dev));
   if (true)
      printDeviceProp(&deviceProp,stderr);

   if (deviceProp.major < 1) {
      fprintf(stderr,"CUDA error: GPU device does not support CUDA.\n");
      exit(-1);
   }

   CUDA_SAFE_CALL(cudaGLSetGLDevice(0));
}

/** Generate vertex arrays for particles.
 *    We need two VAs for flip-flop calculation.
 */
void  particles_genBuffers()
{
   glGenBuffers(2,vbo_pos);
   glGenBuffers(2,nbo_vel);
   glGenBuffers(2,cbo_col);

   // initialize array 0
   glBindVertexArray(va_particles[0]);

      glBindBuffer(GL_ARRAY_BUFFER, vbo_pos[0]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(real)*mNumParticles*4, NULL, GL_DYNAMIC_DRAW);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0,4,GL_REAL,GL_FALSE,0,NULL);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cuda_vbo_pos[0],vbo_pos[0],0));

      glBindBuffer(GL_ARRAY_BUFFER, nbo_vel[0]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(real)*mNumParticles*2, NULL, GL_DYNAMIC_DRAW);
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1,2,GL_REAL,GL_FALSE,0,NULL);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cuda_nbo_vel[0],nbo_vel[0],0));

      glBindBuffer(GL_ARRAY_BUFFER, cbo_col[0]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(float)*mNumParticles*4, NULL, GL_DYNAMIC_DRAW);
      glEnableVertexAttribArray(2);
      glVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE,0,NULL);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cuda_cbo_col[0],cbo_col[0],0));

   glBindVertexArray(0);


   // initialize array 1
   glBindVertexArray(va_particles[1]);

      glBindBuffer(GL_ARRAY_BUFFER, vbo_pos[1]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(real)*mNumParticles*4, NULL, GL_DYNAMIC_DRAW);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0,4,GL_REAL,GL_FALSE,0,NULL);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cuda_vbo_pos[1],vbo_pos[1],0));

      glBindBuffer(GL_ARRAY_BUFFER, nbo_vel[1]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(real)*mNumParticles*2, NULL, GL_DYNAMIC_DRAW);
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1,2,GL_REAL,GL_FALSE,0,NULL);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cuda_nbo_vel[1],nbo_vel[1],0));

      glBindBuffer(GL_ARRAY_BUFFER, cbo_col[1]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(float)*mNumParticles*4, NULL, GL_DYNAMIC_DRAW);
      glEnableVertexAttribArray(2);
      glVertexAttribPointer(2,4,GL_FLOAT,GL_FALSE,0,NULL);
      CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&cuda_cbo_col[1],cbo_col[1],0));

   glBindVertexArray(0);
}

/** Destroy particle buffer objects.
 *
 */
void  particles_destroyBuffers()
{
   if (h_particlePosition!=NULL)
      delete [] h_particlePosition;
   h_particlePosition = NULL;

   if (h_particleVelocity!=NULL)
      delete [] h_particleVelocity;
   h_particleVelocity = NULL;

   if (h_particleColor!=NULL)
      delete [] h_particleColor;
   h_particleColor = NULL;

   if (!mParticleList.empty())
      mParticleList.clear();

   if (cuda_vbo_pos[0]!=NULL) {
      CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cuda_vbo_pos[0]));
      CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cuda_vbo_pos[1]));
   }
   if (cuda_nbo_vel[0]!=NULL) {
      CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cuda_nbo_vel[0]));
      CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cuda_nbo_vel[1]));
   }
   if (cuda_cbo_col[0]!=NULL) {
      CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cuda_cbo_col[0]));
      CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cuda_cbo_col[1]));
   }

   if (vbo_pos[0]>0)
      glDeleteBuffers(2,vbo_pos);
   vbo_pos[0] = vbo_pos[1] = 0;

   if (nbo_vel[0]>0)
      glDeleteBuffers(2,nbo_vel);
   nbo_vel[0] = nbo_vel[1] = 0;

   if (cbo_col[0]>0)
      glDeleteBuffers(2,cbo_col);
   cbo_col[0] = cbo_col[1] = 0;

   if (h_mass!=NULL)
      delete [] h_mass;
   h_mass = NULL;

   if (h_charge!=NULL)
      delete [] h_charge;
   h_charge = NULL;

   if (d_mass!=NULL)
      cudaFree(d_mass);
   d_mass = NULL;

   if (d_charge!=NULL)
      cudaFree(d_charge);
   d_charge = NULL;
}

/** Set mass and charge of the particles
 *   and copy them to the GPU.
 */
void  particles_setMassAndCharge()
{
   assert(h_mass!=NULL);
   assert(h_charge!=NULL);

   if (d_mass>0)   cudaFree(d_mass);
   if (d_charge>0) cudaFree(d_charge);

   size_t memSize = sizeof(real)*mNumParticles;
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_mass,   memSize));
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_charge, memSize));
   CUDA_SAFE_CALL(cudaMemcpy(d_mass,  h_mass,  memSize, cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(d_charge,h_charge,memSize, cudaMemcpyHostToDevice));
}

/** Initialize particle vertex arrays
 *
 */
void  init_Particles( )
{
   particles_destroyBuffers();
   set_Particles();              // implemented in .inl scene files

   if (mNumParticles<=0)
      return;

   // If set, add constant velocity read from command line to all particles.
   if (mAddVelSet) {
      for(unsigned int i=0; i<mNumParticles; i++) {
         h_particleVelocity[2*i+0] += mAddVelocity[0];
         h_particleVelocity[2*i+1] += mAddVelocity[1];
      }
   }

   fprintf(stderr,"Initialize ... %d ... particles.\n",mNumParticles);
   assert(h_particlePosition!=NULL);
   assert(h_particleVelocity!=NULL);

   particles_genBuffers();

   glBindBuffer(GL_ARRAY_BUFFER,vbo_pos[0]);
   glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(real)*mNumParticles*4,h_particlePosition);
   glBindBuffer(GL_ARRAY_BUFFER,0);

   glBindBuffer(GL_ARRAY_BUFFER,nbo_vel[0]);
   glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(real)*mNumParticles*2,h_particleVelocity);
   glBindBuffer(GL_ARRAY_BUFFER,0);

   glBindBuffer(GL_ARRAY_BUFFER,cbo_col[0]);
   glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(float)*mNumParticles*4,h_particleColor);
   glBindBuffer(GL_ARRAY_BUFFER,0);

   particles_setMassAndCharge();
}

/** Allocate host memory for particle data.
 */
void allocHostMemForParticles( )
{
   assert(mNumParticles>0);

   h_particlePosition = new real[mNumParticles*4];
   h_particleVelocity = new real[mNumParticles*2];
   h_particleColor    = new float[mNumParticles*4];

   h_mass     = new real[mNumParticles];
   h_charge   = new real[mNumParticles];

   // Initialize particles...
   for(unsigned int i=0; i<mNumParticles; i++)
   {
      for(unsigned int c=0; c<4; c++) {
         h_particlePosition[c] = (real)0;
         h_particleColor[c]    = 1.0f;
         if (c<2)
            h_particleVelocity[c] = (real)0;
      }
      h_mass[i]   = (real)1;
      h_charge[i] = (real)1;
   }
}

/** Load particles from file.
 *
 *   Particle data is stored in the following order:
 *      number of particle,
 *      object number the particle belongs to,
 *      particle is fixed (1) or free (0),
 *      initial u coordinate,
 *      initial v coordinate,
 *      initial u velocity,
 *      initial v velocity,
 *      particle mass,
 *      particle charge,
 *      color (r,g,b,a)
 *
 * \param filename
 */
void  particles_load(const char* filename)
{
   FILE* fptr = fopen(filename,"r");
   if (fptr==NULL) {
      fprintf(stderr,"Cannot open file %s for reading.\n",filename);
      return;
   }
   fprintf(stderr,"Load particle file: %s\n",filename);

   particles_destroyBuffers();

   fscanf(fptr,"%d",&mNumParticles);
   fprintf(stderr,"# particles: %d\n",mNumParticles);

   ParticleList newParticleList = {0,0,0};
   mParticleList.push_back(newParticleList);

   if (mNumParticles>0)
   {
      h_particlePosition = new real[mNumParticles*4];
      h_particleVelocity = new real[mNumParticles*2];
      h_particleColor    = new float[mNumParticles*4];

      h_mass     = new real[mNumParticles];
      h_charge   = new real[mNumParticles];

      int num;
      double pos[4], vel[2], mass, charge;
      float  col[4];
      int currObjNum = 0;

      for(unsigned int i=0; i<mNumParticles; i++)
      {
         fscanf(fptr,"%d  %lf %lf  %lf %lf  %lf %lf  %lf %lf  %f %f %f %f",&num,
                     &pos[3],&pos[2],&pos[0],&pos[1], &vel[0],&vel[1], &mass, &charge,
                     &col[0],&col[1],&col[2],&col[3]);

         if ((int)pos[3]>currObjNum) {
            newParticleList.num_i   = mParticleList[currObjNum].num_f;
            newParticleList.num_f   = newParticleList.num_i;
            currObjNum++;
            newParticleList.obj_num = currObjNum;
            mParticleList.push_back(newParticleList);
         }
         else {
            mParticleList[currObjNum].num_f++;
         }

         for(int c=0; c<4; c++) {
            h_particlePosition[4*i+c] = (real)pos[c];
            h_particleColor[4*i+c]    = col[c];
         }
         h_particleVelocity[2*i+0] = (real)vel[0] + mAddVelocity[0];
         h_particleVelocity[2*i+1] = (real)vel[1] + mAddVelocity[1];

         h_mass[i]   = (real)mass;
         h_charge[i] = (real)charge;

#if 0
         fprintf(stderr,"%10.6f %10.6f %d %2d  %10.6f %10.6f  %6.2f %6.2f  %4.2f %4.2f %4.2f %4.2f\n",
                  h_particlePosition[4*i+0],h_particlePosition[4*i+1],(int)h_particlePosition[4*i+2],(int)h_particlePosition[4*i+3],
                  h_particleVelocity[2*i+0],h_particleVelocity[2*i+1],
                  h_mass[i],h_charge[i],h_particleColor[0],h_particleColor[1],h_particleColor[2],h_particleColor[3]);
#endif
      }

      fprintf(stderr,"\nParticle list...\n");
      for(unsigned int p=0; p<mParticleList.size(); p++)
         fprintf(stderr,"%d ... %2d %5d %5d\n",p,mParticleList[p].obj_num,mParticleList[p].num_i,mParticleList[p].num_f);

      particles_genBuffers();

      glBindBuffer(GL_ARRAY_BUFFER,vbo_pos[0]);
      glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(real)*mNumParticles*4,h_particlePosition);
      glBindBuffer(GL_ARRAY_BUFFER,0);

      glBindBuffer(GL_ARRAY_BUFFER,nbo_vel[0]);
      glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(real)*mNumParticles*2,h_particleVelocity);
      glBindBuffer(GL_ARRAY_BUFFER,0);

      glBindBuffer(GL_ARRAY_BUFFER,cbo_col[0]);
      glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(float)*mNumParticles*4,h_particleColor);
      glBindBuffer(GL_ARRAY_BUFFER,0);

      particles_setMassAndCharge();

      curr_buf = 0;
      new_buf  = 1;
   }
   fclose(fptr);
   params.numParticles = mNumParticles;
}

/** Save particles from current buffer 0 to file.
 *
 *   Particle data is stored in the following order:
 *      number of particle,
 *      object number the particle belongs to,
 *      particle is fixed (1) or free (0),
 *      initial u coordinate,
 *      initial v coordinate,
 *      initial u velocity,
 *      initial v velocity,
 *      particle mass,
 *      particle charge,
 *      color (r,g,b,a)
 *
 * \param setVelToZero : set velocity to zero before saving.
 */
void particles_save( bool setVelToZero )
{
   fprintf(stderr,"Save particles ... ");
   assert(mNumParticles>0);
   real*  particles  = new real[mNumParticles*4];
   real*  velocities = new real[mNumParticles*2];
   float* colors     = new float[mNumParticles*4];

   glBindBuffer(GL_ARRAY_BUFFER,vbo_pos[0]);
   glGetBufferSubData(GL_ARRAY_BUFFER,0,sizeof(real)*mNumParticles*4,particles);
   glBindBuffer(GL_ARRAY_BUFFER,nbo_vel[0]);
   glGetBufferSubData(GL_ARRAY_BUFFER,0,sizeof(real)*mNumParticles*2,velocities);
   glBindBuffer(GL_ARRAY_BUFFER,cbo_col[0]);
   glGetBufferSubData(GL_ARRAY_BUFFER,0,sizeof(float)*mNumParticles*4,colors);
   glBindBuffer(GL_ARRAY_BUFFER,0);

   std::string filename = "output/" + outFileName;
   FILE* fptr = fopen(filename.c_str(),"w");
   if (fptr!=NULL) {
      fprintf(stderr,"%s ... ",filename.c_str());
      fprintf(fptr,"%d\n",mNumParticles);
      for(unsigned int i=0; i<mNumParticles; i++)
      {
         if (setVelToZero) {
            velocities[2*i+0] = velocities[2*i+1] = (real)0;
         }
         fprintf(fptr,"%4d %2d %d  %10.6f %10.6f %12.6f %12.6f  %6.2f %6.2f  %4.2f %4.2f %4.2f %4.2f\n",i,
                  (int)particles[4*i+3],(int)particles[4*i+2],particles[4*i+0],particles[4*i+1],
                  velocities[2*i+0],velocities[2*i+1],
                  h_mass[i],h_charge[i],
                  colors[4*i+0],colors[4*i+1],colors[4*i+2],colors[4*i+3]);
      }
      fclose(fptr);
      fprintf(stderr,"done.\n");
   }
   else {
      fprintf(stderr,"failed.\n");
   }
   delete [] velocities;
   delete [] particles;
   delete [] colors;
}

/** Calculate 3D particle positions from uv parameters.
 * \param withVelocity : also calculate velocity (f^2)
 */
void  calcParticle3Dpos( bool withVelocity = false )
{
   real4  *d_uv_pos, *d_cart_pos;
   real2  *d_uv_vel;

   // map resources
   CUDA_SAFE_CALL(cudaGraphicsMapResources(2,cuda_vbo_pos,0));
   if (withVelocity)
      CUDA_SAFE_CALL(cudaGraphicsMapResources(2,cuda_nbo_vel,0));

   size_t num_bytes;
   CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_uv_pos,   &num_bytes, cuda_vbo_pos[curr_buf]));
   CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_cart_pos, &num_bytes, cuda_vbo_pos[new_buf]));
   if (withVelocity)
      CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_uv_vel, &num_bytes, cuda_nbo_vel[curr_buf]));

   // map uv to 3d coordinates
   if (withVelocity)
      launch_kernelPosVel(mNumParticles,d_obj_params, d_uv_pos, d_uv_vel, d_cart_pos );
   else
      launch_kernelPositions(mNumParticles,d_obj_params, d_uv_pos, d_cart_pos );
   CUDA_CHECK_ERROR();

   // unmap resources
   CUDA_SAFE_CALL(cudaGraphicsUnmapResources(2,cuda_vbo_pos,0));
   if (withVelocity)
      CUDA_SAFE_CALL(cudaGraphicsUnmapResources(2,cuda_nbo_vel,0));

   // read 3d particle positions from vbo
   assert(h_particlePosition!=NULL);
   glBindBuffer(GL_ARRAY_BUFFER,vbo_pos[new_buf]);
   glGetBufferSubData(GL_ARRAY_BUFFER,0,sizeof(real)*mNumParticles*4,h_particlePosition);
   glBindBuffer(GL_ARRAY_BUFFER,0);
}

/** Initialize field line buffers.
 */
void  init_FieldLines()
{
   glGenVertexArrays(1,&va_fieldLines);
   glGenBuffers(1,&vbo_fieldLines);

   glBindVertexArray(va_fieldLines);

   glBindBuffer(GL_ARRAY_BUFFER, vbo_fieldLines);
   glBufferData(GL_ARRAY_BUFFER, sizeof(real)*mNumParticles*mMaxNumPoints*4, NULL, GL_DYNAMIC_DRAW);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0,4,GL_REAL,GL_FALSE,0,NULL);
   glBindBuffer(GL_ARRAY_BUFFER,0);

   glBindVertexArray(0);
}

/** Calculate total electric field of all charges at specific position.
 * \param  pos : specific position.
 * \param particlePtr : pointer to all particles
 * \param charge : pointer to all charges.
 */
rvec3  calcEField ( rvec3 pos, real* particlePtr, real* charge )
{
   rvec3 electricField  = rvec3(0.0,0.0,0.0);
   for(unsigned int i=0; i<mNumParticles; i++)
   {
      rvec3 relpos = rvec3( pos[0]-particlePtr[4*i+0],
                            pos[1]-particlePtr[4*i+1],
                            pos[2]-particlePtr[4*i+2] );
      real rn = glm::length(relpos);
      real edrn3   = charge[i]/(rn*rn*rn);
      electricField += edrn3*relpos;
   }
   return electricField;
}

/** Calculate field lines.
 *
 */
void  calcFieldlines()
{
   assert(mNumParticles>0);
   fprintf(stderr,"Calculate fieldlines ...\n");

   // read 2d particle parameters from vbo
   real*  particlesUV  = new real[mNumParticles*4];
   glBindBuffer(GL_ARRAY_BUFFER,vbo_pos[curr_buf]);
   glGetBufferSubData(GL_ARRAY_BUFFER,0,sizeof(real)*mNumParticles*4,particlesUV);
   glBindBuffer(GL_ARRAY_BUFFER,0);

#if 0
   // Save uv particle positions...
   std::string filename = "p_uv.dat";
   FILE* fptr = fopen(filename.c_str(),"w");
   if (fptr!=NULL) {
      fprintf(stderr,"%s ... ",filename.c_str());
      for(unsigned int i=0; i<mNumParticles; i++) {
         fprintf(fptr,"%4d %10.6f %10.6f\n",i,particlesUV[4*i+0],particlesUV[4*i+1]);
      }
      fclose(fptr);
   }
#endif

   calcParticle3Dpos();

#if 0
   // Save 3D particle positions...
   filename = "p_3d.dat";
   fptr = fopen(filename.c_str(),"w");
   if (fptr!=NULL) {
      fprintf(stderr,"%s ... ",filename.c_str());
      for(unsigned int i=0; i<mNumParticles; i++) {
         fprintf(fptr,"%4d %10.6f %10.6f %10.6f\n",i,h_particlePosition[4*i+0],h_particlePosition[4*i+1],h_particlePosition[4*i+2]);
      }
      fclose(fptr);
   }
#endif

   // Allocate memory for field line data...
   if (h_numPointsPerFieldLine!=NULL)
      delete [] h_numPointsPerFieldLine;
   if (h_fieldLines!=NULL)
      delete [] h_fieldLines;

   h_numPointsPerFieldLine = new int[mNumParticles];
   h_fieldLines = new real[mNumParticles*mMaxNumPoints*4];
   memset(h_numPointsPerFieldLine,0,mNumParticles*sizeof(int));
   memset(h_fieldLines,0,mNumParticles*mMaxNumPoints*4*sizeof(real));

#ifndef _WIN32
   int64_t t1 = get_system_clock();
#endif
   // for particle pNum do...

   unsigned int pNum = 0;
   for(pNum=0; pNum<mNumParticles; pNum++)
   {
      // particle belongs to obj number 'objNum'
      int objNum  = (int)particlesUV[4*pNum+3];
      // this object is of surface type...
      int surfaceType = mObjectList[objNum].type;

      h_numPointsPerFieldLine[pNum] = mMaxNumPoints;

      // ------------------------------------
      //  The initial position follows from the particle position
      //  plus a small offset in the direction of the normal to
      //  the surface
      // ------------------------------------
      real n[3];
      e_calcNormal(surfaceType,&objParams[objNum],&particlesUV[4*pNum],n);
      rvec3 pos = rvec3(h_particlePosition[4*pNum+0],h_particlePosition[4*pNum+1],h_particlePosition[4*pNum+2])
                + dl * rvec3(n[0],n[1],n[2]);

      rvec3 E = glm::normalize(calcEField(pos,h_particlePosition,h_charge));
      for(int i=0; i<mMaxNumPoints; i++) {
         int num = 4*(pNum*mMaxNumPoints+i);
         h_fieldLines[num+0] = pos[0];
         h_fieldLines[num+1] = pos[1];
         h_fieldLines[num+2] = pos[2];
         h_fieldLines[num+3] = (real)1;
         if (h_charge[pNum]<0)
            pos -= E*dl;
         else
            pos += E*dl;
         E = glm::normalize(calcEField(pos,h_particlePosition,h_charge));
         //if (pNum==0)  printf_vec3(pos,stdout);
      }
      //fprintf(stdout,"\n");
   }
#ifndef _WIN32
   int64_t t2 = get_system_clock();
   std::cerr << (t2-t1)*1e-6 << std::endl;
#endif

   delete [] particlesUV;

   glBindBuffer(GL_ARRAY_BUFFER,vbo_fieldLines);
   glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(real)*mNumParticles*mMaxNumPoints*4,h_fieldLines);
   glBindBuffer(GL_ARRAY_BUFFER,0);

   mShowFLines = true;
}

/** Calculate energies
 *    Distances have to be given in millimeters (mm).
 &
 * \param Ekin : kinetic energy   (atto eV).
 * \param Efield : field energy   (atto eV).
 * \param coc : center of charge  (mm).
 */
void  calcEnergies( real &Ekin, real &Efield, rvec3 &coc )
{
   assert(mNumParticles>0);
   calcParticle3Dpos(true);

   rvec3 ri,rj;
   real ed;
   Ekin   = (real)0;
   Efield = (real)0;

   // center of charge (coc) and total charge (totc)
   coc = rvec3();
   real  totc = (real)0;

   real EPS = 0e-6;

   for(unsigned int i=0; i<mNumParticles; i++)
   {
      Ekin += (real)(0.5*h_mass[i]*h_particlePosition[4*i+3]);

      ri = rvec3(h_particlePosition[4*i+0],h_particlePosition[4*i+1],h_particlePosition[4*i+2]);

      coc  += h_charge[i]*ri;
      totc += h_charge[i];

      //for(unsigned int j=0; j<i; j++) {
      for(unsigned int j=0; j<mNumParticles; j++) {
         rj = rvec3(h_particlePosition[4*j+0],h_particlePosition[4*j+1],h_particlePosition[4*j+2]);
         ed = (real)(1.0/(glm::length(ri-rj)+EPS));
         if (j<i && i>0)
            Efield += h_charge[i]*h_charge[j]*ed;
      }
   }

   coc = coc/totc;
}

/** Draw particles
 * \param pList : reference to particle list.
 */
void draw_Particles( ParticleList &pList )
{
   if (pList.obj_num>=mObjectList.size())
      return;

   Object* obj = &mObjectList[pList.obj_num];

   mShader.setUniformValue("center",(float)obj->center[0],(float)obj->center[1],(float)obj->center[2]);
   mShader.setUniformValue("e1",(float)obj->e1[0],(float)obj->e1[1],(float)obj->e1[2]);
   mShader.setUniformValue("e2",(float)obj->e2[0],(float)obj->e2[1],(float)obj->e2[2]);
   mShader.setUniformValue("e3",(float)obj->e3[0],(float)obj->e3[1],(float)obj->e3[2]);

   char parBuf[7];
   for(unsigned int i=0; i<obj->params.size(); i++)
   {
      sprintf(parBuf,"param%d",i);
      mShader.setUniformValue(std::string(parBuf),(float)obj->params[i]);
   }
   mShader.setUniformValue("objType",obj->type);

   GLCE(glBindVertexArray(va_particles[curr_buf]));
   GLCE(glDrawArrays( GL_POINTS, pList.num_i, pList.num_f-pList.num_i ));
   GLCE(glBindVertexArray(0));
}

/** Initialize vertex array and vertex buffer object for box
 *
 */
void init_Box()
{
   glGenVertexArrays(1,&va_box);
   glGenBuffers(1,&vbo_box);

   glBindVertexArray(va_box);

   glBindBuffer(GL_ARRAY_BUFFER,vbo_box);
      glBufferData(GL_ARRAY_BUFFER,sizeof(GLfloat)*numSimpleBoxVerts*4,simpleBoxVerts,GL_STATIC_DRAW);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE, 0, NULL);
   glBindBuffer(GL_ARRAY_BUFFER,0);

   glBindVertexArray(0);
}


/**  Draw object.
 * \param numObj : object number.
 * \param obj : reference to object.
 */
void draw_Object( unsigned int numObj, Object &obj )
{
   if (mFBOList[numObj]!=0) {
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, mFBOTexList[numObj]);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      mShader.setUniformValue("fboTex",1);
   }
   mShader.setUniformValue("showPoti",0);

#ifdef WITH_POTENTIAL
   if (mFBOpotiList[numObj]!=0 && mShowPoti) {
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, mFBOpotiTexList[numObj]);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
      mShader.setUniformValue("fboPotiTex",1);
      mShader.setUniformValue("showPoti",1);
   }
#endif

   mShader.setUniformValue("objType",obj.type);

   double u_step = (obj.u_range[1] - obj.u_range[0])/(double)obj.slice_u;
   double v_step = (obj.v_range[1] - obj.v_range[0])/(double)obj.slice_v;

   mShader.setUniformValue("slice",  obj.slice_u,obj.slice_v);
   mShader.setUniformValue("uvStep", (float)u_step,(float)v_step);
   mShader.setUniformValue("uv_min", (float)obj.u_range[0], (float)obj.v_range[0]);
   mShader.setUniformValue("uv_max", (float)obj.u_range[1], (float)obj.v_range[1]);
   mShader.setUniformValue("useMod", obj.use_modulo[0], obj.use_modulo[1]);

   mShader.setUniformValue("center",(float)obj.center[0],(float)obj.center[1],(float)obj.center[2]);
   mShader.setUniformValue("e1",(float)obj.e1[0],(float)obj.e1[1],(float)obj.e1[2]);
   mShader.setUniformValue("e2",(float)obj.e2[0],(float)obj.e2[1],(float)obj.e2[2]);
   mShader.setUniformValue("e3",(float)obj.e3[0],(float)obj.e3[1],(float)obj.e3[2]);

   mShader.setUniformValue("showsplats",(int)mParticleVis);

   char parBuf[7];
   for(unsigned int i=0; i<obj.params.size(); i++)
   {
      sprintf(parBuf,"param%d",i);
      mShader.setUniformValue(std::string(parBuf),(float)obj.params[i]);
   }

   if (obj.wireFrame || mShowWired) {
      glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE );
   }
   else {
      glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL );
   }

   int num = obj.slice_u * obj.slice_v;
   glBindVertexArray(va_object);
   glDrawArraysInstanced(GL_TRIANGLE_STRIP,0,4,num);
   glBindVertexArray(0);

   glBindTexture(GL_TEXTURE_2D,0);
}

/** Draw box
 *
 */
void draw_Box()
{
   mShader.setUniformValue("objType",-1);
   mShader.setUniformValue("isSurface",-1);
   mShader.setUniformValue("showsplats",-1);

   mShader.setUniformValue("center",0.0f,0.0f,0.0f);
   mShader.setUniformValue("e1",1.0f,0.0f,0.0f);
   mShader.setUniformValue("e2",0.0f,1.0f,0.0f);
   mShader.setUniformValue("e3",0.0f,0.0f,1.0f);

   mShader.setUniformValue("useFixedColor",1);
   mShader.setUniformValue("fixedColor",0.5f,0.5f,0.3f);
   //glEnable(GL_LINE_SMOOTH);

   glBindVertexArray(va_box);
   glDrawArrays( GL_LINES, 0, numSimpleBoxVerts );
   glBindVertexArray(0);

   mShader.setUniformValue("useFixedColor",0);
}

/** Draw field lines
 *
 */
void  draw_FieldLines()
{
   mShader.setUniformValue("objType",-1);
   mShader.setUniformValue("isSurface",-1);
   mShader.setUniformValue("showsplats",-1);

   mShader.setUniformValue("center",0.0f,0.0f,0.0f);
   mShader.setUniformValue("e1",1.0f,0.0f,0.0f);
   mShader.setUniformValue("e2",0.0f,1.0f,0.0f);
   mShader.setUniformValue("e3",0.0f,0.0f,1.0f);

   mShader.setUniformValue("useFixedColor",1);
   mShader.setUniformValue("fixedColor",0.6f,0.8f,0.6f);

   glEnable(GL_LINE_SMOOTH);
   glLineWidth(1);

   glBindVertexArray(va_fieldLines);
   for(unsigned int i=0; i<mNumParticles; i++) {
      glDrawArrays( GL_LINE_STRIP, i*mMaxNumPoints, mMaxNumPoints );
   }
   glBindVertexArray(0);

   mShader.setUniformValue("useFixedColor",0);
}

/** Draw help.
 *
 */
void draw_Help()
{
   float xPos = 100.0f;
   float yPos = mWindowHeight - 50.0f;
   float yStep = 18.0f;

   float offset = 10.0f;
   float width  = 350.0f + 2*offset;
   float height = 24*yStep + 2*offset;

   glEnable(GL_BLEND);
   glViewport(0,0,mWindowWidth,mWindowHeight);

   glm::mat4 projMX = glm::ortho(0.0f,(float)mWindowWidth,0.0f,(float)mWindowHeight,-1.0f, 1.0f);

   mHelpShader.bind();
   mHelpShader.setUniformValue("bgColor",0.2f,0.2f,0.2f,0.95f);
   mHelpShader.setUniformValue("origin",xPos-offset,yPos+yStep+offset);
   mHelpShader.setUniformValue("size",width,-height);
   glUniformMatrix4fv( mHelpShader.uniformLocation("proj_matrix"), 1, GL_FALSE, glm::value_ptr(projMX) );
   glBindVertexArray(va_object);
     glDrawArrays(GL_TRIANGLE_STRIP,0,4);
   glBindVertexArray(0);
   mHelpShader.release();
   glDisable(GL_BLEND);

   mRenderText->printf(xPos,yPos," b,B : Show/Hide box");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," e   : Calculate energies");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," f   : Show/Hide FBOs");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," F   : Save particle data");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," G   : Save particle data (zero velocity)");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," h   : Show/Hide this help");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," K   : Calculate field lines");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," m   : Toggle particle visualization");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," o,O : Show/Hide objects");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," p,P : Toggle play");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," r   : Reset camera POI");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," R   : Reset/Reload particle data");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," s   : Single kernel call");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," S   : Reload shaders");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," w   : Toggle wireframe");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," v,V : Toggle surface shaders");  yPos-=yStep;
   //mRenderText->printf(xPos,yPos," u   : Calculate potential");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," y   : Save current image");  yPos-=yStep;
   mRenderText->printf(xPos,yPos," 1-9 : Jerk the particles with 2^n");  yPos-=yStep;
   
   yPos-=yStep;
   mRenderText->printf(xPos,yPos," Mouse control:");  yPos-=yStep;
   mRenderText->printf(xPos,yPos,"  left       : rotate around POI"); yPos-=yStep;
   mRenderText->printf(xPos,yPos,"  left+SHIFT : change distance to POI"); yPos-=yStep;
   mRenderText->printf(xPos,yPos,"  mid        : change z-value of POI"); yPos-=yStep;
   mRenderText->printf(xPos,yPos,"  right      : change xy-value of POI"); yPos-=yStep;
}

/** Render to FBO.
 *    Each FBO texture is then mapped onto the corresponding surface.
 */
void  renderToFBO()
{
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

   glViewport(0,0,FBO_TEXTURE_SIZE,FBO_TEXTURE_SIZE);
   glClearColor(0.0f,0.0f,0.0f,0.0f);
   
   for(unsigned int i=0; i<mParticleList.size(); i++)
   {
      int numObj = mParticleList[i].obj_num;
      Object* o = &mObjectList[numObj];

      glBindFramebuffer( GL_FRAMEBUFFER, mFBOList[i] );
      glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

      // The uv-domain of the FBO texture is somewhat bigger than the
      // defined domain  [u_range] x [v_range]

      glm::mat4 projMX = glm::ortho((float)(o->u_range[0] - mFBOoffsetX),
                                    (float)(o->u_range[1] + mFBOoffsetX),
                                    (float)(o->v_range[0] - mFBOoffsetY),
                                    (float)(o->v_range[1] + mFBOoffsetY));

#if 0
      // show fbo borders
      mFBOTestShader.bind();
      glUniformMatrix4fv( mFBOTestShader.uniformLocation("proj_matrix"), 1, GL_FALSE, glm::value_ptr(projMX) );
      mFBOTestShader.setUniformValue("uv_min", (float)o->u_range[0], (float)o->v_range[0]);
      mFBOTestShader.setUniformValue("uv_max", (float)o->u_range[1], (float)o->v_range[1]);
      mFBOTestShader.setUniformValue("offset", (float)mFBOoffsetX,(float)mFBOoffsetY);

      glBindVertexArray(va_object);
      glDrawArrays(GL_TRIANGLE_STRIP,0,4);
      glBindVertexArray(0);

      mFBOTestShader.release();
#endif

      mFBOShader.bind();
      glUniformMatrix4fv( mFBOShader.uniformLocation("proj_matrix"), 1, GL_FALSE, glm::value_ptr(projMX) );
      mFBOShader.setUniformValue("objType",o->type);
      mFBOShader.setUniformValue("ds",mds);
      mFBOShader.setUniformValue("uv_mod",(float)(o->uv_mod[0]),(float)(o->uv_mod[1]));
      mFBOShader.setUniformValue("useMod",o->use_modulo[0],o->use_modulo[1]);

      char parBuf[7];
      for(unsigned int j=0; j<o->params.size(); j++)
      {
         sprintf(parBuf,"param%d",j);
         mFBOShader.setUniformValue(std::string(parBuf),(float)o->params[j]);
      }

      glBindVertexArray(va_particles[curr_buf]);
      glDrawArrays(GL_POINTS, mParticleList[i].num_i, mParticleList[i].num_f-mParticleList[i].num_i );
      glBindVertexArray(0);

      mFBOShader.release();

      glBindFramebuffer( GL_FRAMEBUFFER, 0 );
   }

   glDisable(GL_BLEND);
}

/** Render potential of all particles
 *
 */
#ifdef WITH_POTENTIAL
void renderPotential()
{
   mPlay = false;
   // ------------------------------------------
   //  calculate particle position in 3D
   // ------------------------------------------
   calcParticle3Dpos();

   // ------------------------------------------
   //  calculate potential on surface 'i'
   // ------------------------------------------
   glEnable(GL_BLEND);
   //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   glBlendFunc(GL_SRC_ALPHA,GL_ONE);

   glViewport(0,0,FBO_POTI_TEXTURE_SIZE,FBO_POTI_TEXTURE_SIZE);
   for(unsigned int i=0; i<mParticleList.size(); i++)
   {
      int numObj = mParticleList[i].obj_num;
      Object* o = &mObjectList[numObj];

      glBindFramebuffer( GL_FRAMEBUFFER, mFBOpotiList[i] );
      glClearColor(0.0f,0.0f,0.0f,0.0f);
      glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

      // The uv-domain of the FBO texture is somewhat bigger than the
      // defined domain  [u_range] x [v_range]

      glm::mat4 projMX = glm::ortho(0.0f,1.0f,0.0f,1.0f);

      mFBOPotentialShader.bind();
      glUniformMatrix4fv( mFBOPotentialShader.uniformLocation("proj_matrix"), 1, GL_FALSE, glm::value_ptr(projMX) );
      mFBOPotentialShader.setUniformValue("objType",o->type);
      mFBOPotentialShader.setUniformValue("uv_min", (float)o->u_range[0], (float)o->v_range[0]);
      mFBOPotentialShader.setUniformValue("uv_max", (float)o->u_range[1], (float)o->v_range[1]);

      mFBOPotentialShader.setUniformValue("center",(float)o->center[0],(float)o->center[1],(float)o->center[2]);
      mFBOPotentialShader.setUniformValue("e1",(float)o->e1[0],(float)o->e1[1],(float)o->e1[2]);
      mFBOPotentialShader.setUniformValue("e2",(float)o->e2[0],(float)o->e2[1],(float)o->e2[2]);
      mFBOPotentialShader.setUniformValue("e3",(float)o->e3[0],(float)o->e3[1],(float)o->e3[2]);

      char parBuf[7];
      for(unsigned int j=0; j<o->params.size(); j++) {
         sprintf(parBuf,"param%d",j);
         mFBOPotentialShader.setUniformValue(std::string(parBuf),(float)o->params[j]);
      }

      for(unsigned int p=0; p<mNumParticles; p++) {
         //fprintf(stderr,"%d %f %f %f\n",p,h_particlePosition[4*p+0],h_particlePosition[4*p+1],h_particlePosition[4*p+2]);
         mFBOPotentialShader.setUniformValue("particle_pos",(float)h_particlePosition[4*p+0],(float)h_particlePosition[4*p+1],(float)h_particlePosition[4*p+2]);
         mFBOPotentialShader.setUniformValue("particle_q",(float)h_charge[p]);
         glBindVertexArray(va_object);
         glDrawArrays(GL_TRIANGLE_STRIP,0,4);
         glBindVertexArray(0);
      }

      mFBOPotentialShader.release();
      glBindFramebuffer( GL_FRAMEBUFFER, 0 );
   }
   glDisable(GL_BLEND);
   mShowPoti = true;
}
#endif

/** Draw only FBO for an individual surface.
 *  \param numObj : object number
 */
void draw_FBO(unsigned int numObj)
{
   if (numObj>=mObjectList.size())
      return;

   glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
   glViewport(0,0,mWindowWidth,mWindowHeight);

   glm::mat4 projMX = glm::ortho(0.0f,1.0f,0.0f,1.0f);

   mFBOShowShader.bind();
   if (mFBOList[numObj]!=0) {
      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, mFBOTexList[numObj]);
      mFBOShowShader.setUniformValue("tex",1);
   }

   glUniformMatrix4fv( mFBOShowShader.uniformLocation("proj_matrix"), 1, GL_FALSE, glm::value_ptr(projMX) );

   glBindVertexArray(va_object);
   glDrawArrays(GL_TRIANGLE_STRIP,0,4);
   glBindVertexArray(0);

   mFBOShowShader.release();
}


/** Run CUDA kernel.
 *
 */
void runKernel( bool test = false )
{
   if (mLogEnergy && logEfilePtr!=NULL && (mCountSteps % mLogSteps == 0)) {
      rvec3 coc;
      calcEnergies(mCurrEkin,mCurrEfield,coc);

      //fprintf(logEfilePtr,"%10.6f %13.6e %13.6e   %7.3e %7.3e %7.3e\n",t_curr,mCurrEkin,mCurrEfield,coc.x,coc.y,coc.z);
      fprintf(logEfilePtr,"%10.6f %16.9e %16.9e\n",t_curr,mCurrEkin,mCurrEfield);
   }

   real4  *d_curr_pos, *d_new_pos;
   real2  *d_curr_vel, *d_new_vel;
   float4 *d_curr_col, *d_new_col;

   // map resources
   CUDA_SAFE_CALL(cudaGraphicsMapResources(2,cuda_vbo_pos,0));
   CUDA_SAFE_CALL(cudaGraphicsMapResources(2,cuda_nbo_vel,0));
   CUDA_SAFE_CALL(cudaGraphicsMapResources(2,cuda_cbo_col,0));

   size_t num_bytes;
   CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_curr_pos, &num_bytes, cuda_vbo_pos[curr_buf]));
   CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_curr_vel, &num_bytes, cuda_nbo_vel[curr_buf]));
   CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_curr_col, &num_bytes, cuda_cbo_col[curr_buf]));

   CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_new_pos,  &num_bytes, cuda_vbo_pos[new_buf]));
   CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_new_vel,  &num_bytes, cuda_nbo_vel[new_buf]));
   CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_new_col,  &num_bytes, cuda_cbo_col[new_buf]));

   launch_kernel(mNumParticles,d_params,d_obj_params, d_curr_pos, d_curr_vel, d_curr_col,
                                                      d_new_pos,  d_new_vel,  d_new_col,
                                                      d_mass, d_charge );
   CUDA_CHECK_ERROR();

   // unmap resources
   CUDA_SAFE_CALL(cudaGraphicsUnmapResources(2,cuda_cbo_col,0));
   CUDA_SAFE_CALL(cudaGraphicsUnmapResources(2,cuda_nbo_vel,0));
   CUDA_SAFE_CALL(cudaGraphicsUnmapResources(2,cuda_vbo_pos,0));

   std::swap(curr_buf,new_buf);
   if (!test)
      renderToFBO();
   t_curr += params.hStep;
   mCountSteps++;
}

/** Do jerk to each particle
 * \param factor
 */
void doJerk( double factor )
{
   real2 *d_curr_vel;
   CUDA_SAFE_CALL(cudaGraphicsMapResources(2,cuda_nbo_vel,0));

   size_t num_bytes;
   CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)&d_curr_vel, &num_bytes, cuda_nbo_vel[curr_buf]));

   launch_kernelJerk(mNumParticles, d_curr_vel, (real)factor);

   CUDA_CHECK_ERROR();
   CUDA_SAFE_CALL(cudaGraphicsUnmapResources(2,cuda_nbo_vel,0));
}

/**  display callback function
 */
void  cb_display()
{
   glClearColor(bgColor[0],bgColor[1],bgColor[2],0.0f);
   glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

   glViewport(0,0,mWindowWidth,mWindowHeight);
   glm::mat4 viewMX = mCamera.viewMatrix();

   mShader.bind();
   glUniformMatrix4fv( mShader.uniformLocation("proj_matrix"), 1, GL_FALSE, mCamera.projMatrixPtr() );

   glm::vec3 obsPos = mCamera.getEyePos();
   mShader.setUniformValue("obsPos",(float)obsPos.x,(float)obsPos.y,(float)obsPos.z);

// --- draw box ---
   if (mShowBox) {
      viewMX = glm::translate(viewMX,glm::vec3(-0.5f));
      glUniformMatrix4fv( mShader.uniformLocation("view_matrix"), 1, GL_FALSE, glm::value_ptr(viewMX) );
      draw_Box();
   }

   viewMX = mCamera.viewMatrix();
   glUniformMatrix4fv( mShader.uniformLocation("view_matrix"), 1, GL_FALSE, glm::value_ptr(viewMX) );

// --- draw objects ---
#if 1
   //glEnable(GL_CULL_FACE);
   //glCullFace(GL_BACK);

   if (mShowObjs) {
      mShader.setUniformValue("isSurface",1);
      mShader.setUniformValue("whichShading",mWhichShading);
      mShader.setUniformValue("fboOffset",   mFBOoffsetX,mFBOoffsetY);
      for(unsigned int i=0; i<mObjectList.size(); i++) {
         draw_Object(i,mObjectList[i]);
      }
   }
   //glDisable(GL_CULL_FACE);
#endif


// --- draw particles ---
#if 1
   if (mParticleVis==e_particleVis_points_only || mParticleVis==e_particleVis_points_and_splats) {
      mShader.setUniformValue("isSurface",0);
      for(unsigned int i=0; i<mParticleList.size(); i++) {
         draw_Particles(mParticleList[i]);
      }
   }
#endif

   if (h_fieldLines!=NULL && mShowFLines)
      draw_FieldLines();

   mShader.release();

   glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
   if (mShowFBO>=0 && mShowFBO<(int)mObjectList.size())
      draw_FBO(mShowFBO);

   if (mLogEnergy) {
      mRenderText->printf(5.0f,5.0f,"t: %9.5f   T:%16.9e   W:%16.9e\n",t_curr,mCurrEkin,mCurrEfield);
   }
   else {
      mRenderText->printf(5.0f,5.0f,"t: %9.5f\n",t_curr);
   }

   if (mShowHelp && !mRenderImageSeq)
      draw_Help();

   if (!mRenderImageSeq)
      glutSwapBuffers();
}

/** Write ppm image.
 * \param filename : file name
 * \param img : pointer to image data given as unsigned char (rgb)
 */
void writePPM( const char* filename, const unsigned char* img,
               const unsigned int width, const unsigned int height )
{
   //int64_t t1 = get_system_clock();

   FILE* fptr = fopen(filename,"wb");
   if (fptr==NULL)
      return;

   fprintf(fptr,"P6\n%d %d\n255\n",width,height);
#if 1
   for(unsigned int y=0; y<height; y++) {
      fwrite(&img[3*(height-y-1)*width],sizeof(unsigned char),width*3,fptr);
   }
#else
   fwrite(&img[0],sizeof(unsigned char),width*height*3,fptr);
#endif
   fflush(fptr);
   fclose(fptr);

   //int64_t t2 = get_system_clock();
   //fprintf(stderr,"   dt: %f\n",(t2-t1)*1e-3);
}


/** Render image sequence for output
 * \param N : number of images
 * \param stepOver : number of kernel calls before next image output
 */
void renderImageSequence( int N = 1, int stepOver = 0 )
{
   unsigned char* imgBuffer = new unsigned char[mWindowWidth*mWindowHeight*3];

   mRenderImageSeq = true;
   for(int n=0; n<N; n++)
   {
      fprintf(stderr,"\rImage # %3d/%3d",n+1,N);
      glBindFramebuffer( GL_FRAMEBUFFER, mFBOoutput );
      cb_display();
      glBindFramebuffer( GL_FRAMEBUFFER, 0 );

/*
      glReadBuffer(GL_COLOR_ATTACHMENT0);
      glReadPixels(0,0,mWindowWidth,mWindowHeight, GL_RGB, GL_UNSIGNED_BYTE, imgBuffer);
*/
      
      GLCE(glBindTexture(GL_TEXTURE_2D,mFBOtexOutput));
      GLCE(glGetTexImage(GL_TEXTURE_2D,0,GL_RGB,GL_UNSIGNED_BYTE,imgBuffer));
      GLCE(glBindTexture(GL_TEXTURE_2D,0));

      char sbuf[256];
      //sprintf(sbuf,"output/image_t%7.5e_n%05d.ppm",t_curr,n);
      sprintf(sbuf,"output/image_n%05d.ppm",n);
      writePPM(sbuf,imgBuffer,mWindowWidth,mWindowHeight);

      int c = 0;
      while(c<stepOver) {
         runKernel();
         c++;
      }
   }
   mRenderImageSeq = false;
   fprintf(stderr,"\n");

   delete [] imgBuffer;
   glutPostRedisplay();
}

/** reshape callback function
 * \param w : new width
 * \param h : new height
 */
void cb_reshape( int w, int h )
{
   mWindowWidth  = (GLsizei)w;
   mWindowHeight = (GLsizei)h;
   glViewport(0,0,mWindowWidth,mWindowHeight);
   mCamera.setSizeAndAspect(mWindowWidth,mWindowHeight);

   init_FBOoutput();
   fprintf(stderr,"Window: %4d %4d\n",w,h);
}

/** idle callback function
 */
void cb_idle(void)
{
   if (mPlay) {
      runKernel();
      glutPostRedisplay();
   }
}

/** mouse button callback function.
 * \param button : button id
 * \param state  : button state (GLUT_DOWN,GLUT_UP)
 * \param x : current mouse position in x
 * \param y : current mouse position in y
 */
void  cb_handle_mouse( int button, int state, int x, int y )
{
   if (state==GLUT_DOWN) {
      buttonPressed_ = button;
      xlast_ = x;
      ylast_ = y;
   }
   else {
      buttonPressed_ = -1;
   }
}

/** mouse motion callback function.
 * \param x : current mouse position in x
 * \param y : current mouse position in y
 */
void cb_handle_mouse_motion( int x, int y )
{
   int dx = x-xlast_;
   int dy = y-ylast_;
   switch (buttonPressed_) {
      default:
         break;
      case GLUT_LEFT_BUTTON: {
         if (glutGetModifiers() & GLUT_ACTIVE_SHIFT)
            mCamera.changeDistance(dy*mCamera.getDistFactor());
         else
            mCamera.moveOnSphere(dy*mCamera.getRotFactor(),dx*mCamera.getRotFactor());
         break;
      }
      case GLUT_MIDDLE_BUTTON:
         mCamera.moveOnZ(dy*mCamera.getPanXYFactor());
         break;
      case GLUT_RIGHT_BUTTON:
         mCamera.moveOnXY(dy*mCamera.getPanXYFactor(),-dx*mCamera.getPanXYFactor());
         break;
   }
   xlast_ = x;
   ylast_ = y;

#if 1
   // Show camera parameters on stdout...
   if (buttonPressed_>=0) {
      glm::vec3 eye = mCamera.getEyePos();
      glm::vec3 poi = mCamera.getPOI();
      float     dist = mCamera.getDistance();
      float     phi  = mCamera.getPhi();
      float   theta  = mCamera.getTheta();
      fprintf(stdout,"Eye: %7.3f %7.3f %7.3f  POI: %7.3f %7.3f %7.3f   dist:%7.3f phi:%7.3f theta:%7.3f\n",
                      eye.x,eye.y,eye.z,poi.x,poi.y,poi.z,dist,phi,theta);
   }
#endif
   glutPostRedisplay();
}

/** close callback function
 *
 */
void cb_close()
{
   cudaFree(d_obj_params);
   cudaFree(d_params);

   particles_destroyBuffers();
   // delete textures, fbo,...

   if (mFBOdepthOutput!=0)
      glDeleteRenderbuffers(1,&mFBOdepthOutput);

   if (logEfilePtr!=NULL)
      fclose(logEfilePtr);

   std::cerr << "Bye bye...\n";
   exit( EXIT_SUCCESS );
}

/** keyboard callback function
 *
 */
void cb_keyboard( unsigned char key, int x, int y )
{
   if (key>=49 && key<=57) {
      double n = int(key)-48;
      doJerk(pow(2.0,n));
   }

   switch( key ) {
      case 27: {                          //!<  exit program
         glutLeaveMainLoop();
         break;
      }
      case 'b': case 'B': {               //!<  show/hide box
         mShowBox = !mShowBox;
         glutPostRedisplay();
         break;
      }
      case 'e': case 'E': {               //!<  calculate field energy
         rvec3 coc;
         calcEnergies(mCurrEkin,mCurrEfield,coc);
         glutPostRedisplay();
         break;
      }
      case 'f': {                         //!<  show fbo's
         mShowFBO++;
         if (mShowFBO>=(int)mObjectList.size())
            mShowFBO=-1;
         glutPostRedisplay();
         break;
      }
      case 'F': {                         //!<  save particles to file
         particles_save(false);
         break;
      }
      case 'G': {                         //!<  save particles to file with zero velocity
         particles_save(true);
         break;
      }
      case 'h': {                         //!<  show/hide help
         mShowHelp = !mShowHelp;
         glutPostRedisplay();
         break;
      }
      case 'K': {                         //!<  calculate field lines
         mPlay = false;
         calcFieldlines();
         glutPostRedisplay();
         break;
      }
      case 'm': {                         //!<  toggle between particle visualization
         int pt = (int)mParticleVis + 1;
         if (pt>2)
            pt = 0;
         mParticleVis = (e_particleVisType)pt;
         glutPostRedisplay();
         break;
      }
      case 'o': case 'O': {               //!<  show/hide objects
         mShowObjs = !mShowObjs;
         glutPostRedisplay();
         break;
      }
      case 'p': case 'P': {               //!<  play/pause
         mPlay = !mPlay;
         mShowPoti = false;
         mShowFLines = false;
         break;
      }
      case 'r':  {                        //!<  reset camera's poi to origin
         mCamera.setPOI(0,0,0);
         glutPostRedisplay();
         break;
      }
      case 'R': {                         //!<  reset particles using initialization function defined in scene or reload
         bool doPlay = mPlay;
         mPlay = false;
         t_curr = 0.0;
         mCountSteps = 0;
         if (logEfilePtr!=NULL) {
            fclose(logEfilePtr);
            logEfilePtr = fopen(logEfileName.c_str(),"w");
         }
         curr_buf = 0; new_buf = 1;
         if (inFileName!=std::string())
            particles_load(inFileName.c_str());
         else
            init_Particles();

         rvec3 coc;
         calcEnergies(mCurrEkin,mCurrEfield,coc);
         renderToFBO();
         mShowPoti = false;
         mShowFLines = false;
         glutPostRedisplay();
         mPlay = doPlay;
         break;
      }
      case 's': {                         //!<  single kernel call / time step
         runKernel();
         mShowPoti = false;
         mShowFLines = false;
         glutPostRedisplay();
         break;
      }
      case 'S': {                         //!<  reload all shaders
         fprintf(stderr,"Reload shaders...\n");
         mShader.reload();
         mFBOShader.reload();
         mFBOTestShader.reload();
         mFBOShowShader.reload();
         mFBOPotentialShader.reload();
         renderToFBO();
         glutPostRedisplay();
         break;
      }
#ifdef WITH_POTENTIAL      
      case 'u': {
         renderPotential();
         glutPostRedisplay();
         break;
      }
#endif      
      case 'v': {                         //!<  toggle between surface shaders
         mWhichShading++;
         if (mWhichShading>=maxShadingTypes)
            mWhichShading = 0;
         glutPostRedisplay();
         break;
      }
      case 'V': {                         //!<  toggle between surface shaders
         mWhichShading--;
         if (mWhichShading<0)
            mWhichShading = maxShadingTypes-1;
         glutPostRedisplay();
         break;
      }
      case 'w': {                         //!<  show objects as wireframe
         mShowWired = !mShowWired;
         glutPostRedisplay();
         break;
      }
      case 'y': {                         //!<  save image to file
         if ((mWindowWidth % 2 == 0) && (mWindowHeight % 2 == 0)) {
            //renderImageSequence();
            renderImageSequence(500,1);
         }
         break;
      }
#ifndef _WIN32
      case 'T': {                         //!< do a performance test
         int nSteps = 100;
         int64_t t1 = get_system_clock();
         for(int i=0; i<nSteps; i++) {
            fprintf(stderr,"\rn = %3d",i);
            #if 1
               runKernel(false);
            #else
               runKernel();
               cudaThreadSynchronize();
            #endif
         }
         int64_t t2 = get_system_clock();
         fprintf(stderr," ... dt = %.3lf ms per cuda call.\n",(double)(t2-t1)*1e-3/(double)nSteps);
         runKernel();
         glutPostRedisplay();
         break;
      }
#endif
   }
}

/** special keys callback function
 * \param key : special key id
 */
void cb_specialKeys(int key, int , int  )
{
   switch (key) {
      case GLUT_KEY_LEFT:
         mCamera.panning(-mCamera.getPanStep(),0.0f);
         break;
      case GLUT_KEY_RIGHT:
         mCamera.panning(mCamera.getPanStep(),0.0f);
         break;
      case GLUT_KEY_UP:
         mCamera.panning(0.0f,mCamera.getPanStep());
         break;
      case GLUT_KEY_DOWN:
         mCamera.panning(0.0f,-mCamera.getPanStep());
         break;
      case GLUT_KEY_PAGE_DOWN:
         mCamera.changeDistance(mCamera.getDistFactor());
         break;
      case GLUT_KEY_PAGE_UP:
         mCamera.changeDistance(-mCamera.getDistFactor());
         break;
   }
   glutPostRedisplay();
}

/** Read command line arguments.
 * \param argc: number of arguments
 * \param argv: pointer to command line arguments
 */
void  readCmdLine( int argc, char* argv[] )
{
   if (argc<=1)
      return;

   for(int nArg=1; nArg<argc; nArg++)
   {
      if (strcmp(argv[nArg],"-h")==0) {
         fprintf(stderr,"========================================================\n");
         fprintf(stderr,"Command line parameters:\n");
         fprintf(stderr,"\t -dt   <float>   : time step\n");
         fprintf(stderr,"\t -eta  <float>   : frictional constant (>=0)\n");
         fprintf(stderr,"\t -rdamp <float>  : reflection dumping factor (default: 1)\n");
         fprintf(stderr,"\t -load  <string> : load particle positions\n");
         fprintf(stderr,"\t -ds <float>     : point size of particle\n");
         fprintf(stderr,"\t -log            : write energies to logfile\n");
         fprintf(stderr,"\t -E <float> <float> <float> : electric field\n");
         fprintf(stderr,"\t -B <float> <float> <float> : magnetic field\n");
         fprintf(stderr,"\t -v <float> <float> : add u^1, u^2 velocity to all particles\n");
         fprintf(stderr,"========================================================\n");
         exit(1);
      }
      else if (strcmp(argv[nArg],"-dt")==0 && (nArg+1<argc)) {
         params.hStep = (real)atof(argv[nArg+1]);
      }
      else if (strcmp(argv[nArg],"-eta")==0 && (nArg+1<argc)) {
         params.damp  = (real)atof(argv[nArg+1]);
      }
      else if (strcmp(argv[nArg],"-rdamp")==0 && (nArg+1<argc)) {
         params.velReflDamp = (real)atof(argv[nArg+1]);
      }
      else if (strcmp(argv[nArg],"-E")==0 && (nArg+3<argc)) {
         params.E[0] = (real)atof(argv[nArg+1]);
         params.E[1] = (real)atof(argv[nArg+2]);
         params.E[2] = (real)atof(argv[nArg+3]);
      }
      else if (strcmp(argv[nArg],"-B")==0 && (nArg+3<argc)) {
         params.B[0] = (real)atof(argv[nArg+1]);
         params.B[1] = (real)atof(argv[nArg+2]);
         params.B[2] = (real)atof(argv[nArg+3]);
      }
      else if (strcmp(argv[nArg],"-v")==0 && (nArg+2<argc)) {
         mAddVelocity[0] = (real)atof(argv[nArg+1]);
         mAddVelocity[1] = (real)atof(argv[nArg+2]);
         mAddVelSet = true;
      }
      else if (strcmp(argv[nArg],"-load")==0 && (nArg+1<argc)) {
         inFileName = std::string(argv[nArg+1]);
      }
      else if (strcmp(argv[nArg],"-ds")==0 && (nArg+1<argc)) {
         mds = (float)atof(argv[nArg+1]);
      }
      else if (strcmp(argv[nArg],"-log")==0) {
         mLogEnergy = true;
      }
   }

   if (inFileName!=std::string())
      particles_load(inFileName.c_str());
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

/**  This is the main function of the program.
 * \param argc : number of command line arguments
 * \param argv : pointer to command line arguments
 */
int main( int argc, char* argv[] )
{
   // -----------------------------------------------
   //  Initialize glut window
   // -----------------------------------------------
   glutInit(&argc,argv);
   glutInitDisplayMode(GLUT_RGBA|GLUT_ALPHA|GLUT_DOUBLE);
   glutInitContextVersion(3,3);
   glutInitWindowSize(mWindowWidth,mWindowHeight);
   glutCreateWindow("Charged particles on curved surfaces");

   // -----------------------------------------------
   //  Set glut callback functions
   // -----------------------------------------------
   glutDisplayFunc ( cb_display );
   glutReshapeFunc ( cb_reshape );
   glutIdleFunc    ( cb_idle );
   glutMouseFunc   ( cb_handle_mouse );
   glutMotionFunc  ( cb_handle_mouse_motion );
   glutCloseFunc   ( cb_close );
   glutKeyboardFunc( cb_keyboard );
   glutSpecialFunc ( cb_specialKeys );

   // -----------------------------------------------
   //  Initialize OpenGL and load Shaders
   // -----------------------------------------------
   init_OpenGL();
   if (!mShader.createProgram(vShaderFileName,fShaderFileName)) {
      fprintf(stderr,"Cannot create curved surface shader.\n");
      return -1;
   }
   if (!mHelpShader.createProgram(vHelpShaderFileName,fHelpShaderFileName)) {
      fprintf(stderr,"Cannot create help shader.\n");
      return -1;
   }
   if (!mFBOShader.createProgram(vFBOShaderFileName,gFBOShaderFileName,fFBOShaderFileName,GL_POINTS,GL_TRIANGLE_STRIP,4)) {
      fprintf(stderr,"Cannot create FBO shader.\n");
      return -1;
   }
   if (!mFBOTestShader.createProgram(vFBOtestShaderFileName,fFBOtestShaderFileName)) {
      fprintf(stderr,"Cannot create FBO test shader.\n");
      return -1;
   }
   if (!mFBOShowShader.createProgram(vFBOshowShaderFileName,fFBOshowShaderFileName)) {
      fprintf(stderr,"Cannot create FBO show shader.\n");
      return -1;
   }
   if (!mFBOPotentialShader.createProgram(vFBOpotShaderFileName,fFBOpotShaderFileName)) {
      fprintf(stderr,"Cannot create FBO potential shader.\n");
      return -1;
   }

   // -----------------------------------------------
   //  Initialize scene objects and FBOs
   // -----------------------------------------------
   init_Box();
   init_Objects();  // see .inl scene files
   init_FBOs();

#ifdef WITH_POTENTIAL   
   init_FBOpotis();
#endif

   // -----------------------------------------------
   //  Initialize CUDA and particles
   // -----------------------------------------------
   init_CUDA(true);
   init_Particles();

   // -----------------------------------------------
   //  Initialize scene parameters
   // -----------------------------------------------
   params.numParticles = mNumParticles;
   params.damp  = m_damp;
   params.hStep = t_step;
   params.velReflDamp = (real)1;

   // Initially there is neither an external electric nor a magnetic field.
   params.E[0] = params.E[1] = params.E[2] = (real)0;
   params.B[0] = params.B[1] = params.B[2] = (real)0;

   // -----------------------------------------------
   //  Read command line params for step size,
   //  frictional constant, ...
   // -----------------------------------------------
   readCmdLine(argc,argv);

   fprintf(stderr,"\nTime step:       %g\n",params.hStep);
   fprintf(stderr,"Frict. constant: %f\n",params.damp);
   fprintf(stderr,"ReflDamp factor: %f\n",params.velReflDamp);
   fprintf(stderr,"Add velocity:    %f %f\n",mAddVelocity[0],mAddVelocity[1]);
   fprintf(stderr,"E field:         %f %f %f\n",params.E[0],params.E[1],params.E[2]);
   fprintf(stderr,"B field:         %f %f %f\n",params.B[0],params.B[1],params.B[2]);
   fprintf(stderr,"\n");

   // -----------------------------------------------
   //  Allocate CUDA memory and upload particle
   //  parameters
   // -----------------------------------------------
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_params, sizeof(CSparams)));
   CUDA_SAFE_CALL(cudaMemcpy(d_params,&params,sizeof(CSparams), cudaMemcpyHostToDevice));

   assert(mObjectList.size()>0);
   objParams = new CSObjParams[mObjectList.size()];
   for(unsigned int i=0; i<mObjectList.size(); i++)
   {
      objParams[i].obj_type = (int)mObjectList[i].type;

      for(int c=0; c<3; c++) {
         objParams[i].center[c] = mObjectList[i].center[c];
         objParams[i].e1[c] = mObjectList[i].e1[c];
         objParams[i].e2[c] = mObjectList[i].e2[c];
         objParams[i].e3[c] = mObjectList[i].e3[c];
      }

      objParams[i].u_range[0] = mObjectList[i].u_range[0];
      objParams[i].u_range[1] = mObjectList[i].u_range[1];
      objParams[i].v_range[0] = mObjectList[i].v_range[0];
      objParams[i].v_range[1] = mObjectList[i].v_range[1];
      objParams[i].u_mod      = mObjectList[i].uv_mod[0];
      objParams[i].v_mod      = mObjectList[i].uv_mod[1];
      objParams[i].use_modulo[0] = (int)mObjectList[i].use_modulo[0];
      objParams[i].use_modulo[1] = (int)mObjectList[i].use_modulo[1];

      for(unsigned int j=0; j<mObjectList[i].params.size() && j<MAX_NUM_OBJ_PARAMS; j++) {
         objParams[i].value[j] = (real)mObjectList[i].params[j];
      }
   }

   // -----------------------------------------------
   //  Allocate CUDA memory and upload object
   //  parameters
   // -----------------------------------------------
   size_t memsize = sizeof(CSObjParams)*mObjectList.size();
   CUDA_SAFE_CALL(cudaMalloc((void**)&d_obj_params, memsize));
   CUDA_SAFE_CALL(cudaMemcpy(d_obj_params,objParams,memsize, cudaMemcpyHostToDevice));

   // -----------------------------------------------
   //  Initialize camera and field lines
   // -----------------------------------------------
   mCamera.setSizeAndAspect(mWindowWidth,mWindowHeight);
   mCamera.set(glm::vec3(4.0,0.0,0.0),glm::vec3(-1.0,0.0,0.0),glm::vec3(0.0,0.0,1.0));
   mCamera.setFovY(30.0);

   // Set supplement stuff... (defined in the .inl scene files)
   set_Supplement();

   // Initialize field lines...
   init_FieldLines();

   // -----------------------------------------------
   //   Open log file if desired
   // -----------------------------------------------
   if (mLogEnergy) {
      logEfilePtr = fopen(logEfileName.c_str(),"w");
   }
   rvec3 coc;
   calcEnergies(mCurrEkin,mCurrEfield,coc);

   // -----------------------------------------------
   //   Render particle positions onto their
   //   surface
   // -----------------------------------------------
   renderToFBO();

   // -----------------------------------------------
   //   Start GLUT main loop
   // -----------------------------------------------
   glutMainLoop();
   return 0;
}
