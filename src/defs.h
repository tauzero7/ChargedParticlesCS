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

#ifndef DEFS_H
#define DEFS_H


#include <iostream>
#include <vector>
#include <cmath>

extern "C" {
   #include <GL3/gl3w.h>
}

#include <glm/glm.hpp>
#include <glm/core/type.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>


#ifdef _WIN32
   #pragma warning(disable: 4251 4273 4049 4996)
#else
   #include <GL/gl.h>
#endif


#ifdef _WIN32
   #ifndef M_PI   
      #define  M_PI    3.141592653589793
      #define  M_PI_2  1.570796326794897
   #endif  // M_PI
   typedef unsigned char  uint8_t;
#endif  // _WIN32

/* -----------------------------------
 *   global definitions
 * ----------------------------------- */
#ifndef DEG_TO_RAD
   #define DEG_TO_RAD  0.017453292519943295770
   #define RAD_TO_DEG  57.295779513082320875
#endif

#ifndef DBL_MAX
   #define DBL_MAX 1.844674407370955616e19
#endif


/* -----------------------------------
 *   kappa = e^2 / (4*pi*eps_0*m_e)   in m^3/s^2
 *
 *   e     = 1.602176565e-19 C            -> http://physics.nist.gov/cgi-bin/cuu/Value?e|search_for=electron+charge
 *   eps_0 = 1/(mu_0*c^2) C^2/(Jm)        -> http://physics.nist.gov/cgi-bin/cuu/Value?ep0|search_for=dielectric
 *   mu_0  = 4*pi*1e-7 Vs/(Am) (exact)    -> http://physics.nist.gov/cgi-bin/cuu/Value?mu0|search_for=magnetic
 *   c     = 299792458 m/s     (exact)
 *   m_e   = 9.109e-31 kg
 * ----------------------------------- */
#define KAPPA  253.2745



const static float quadVerts[] = {
  0.0f, 0.0f, 0.0f, 1.0f,
  0.0f, 1.0f, 0.0f, 1.0f,
  1.0f, 0.0f, 0.0f, 1.0f,
  1.0f, 1.0f, 0.0f, 1.0f
};

const static float quadTexCoords[] = {
  0.0f, 0.0f,
  1.0f, 0.0f,
  0.0f, 1.0f,
  1.0f, 1.0f
};


const static int   numBoxVerts = 8;
const static float boxVerts[] = {
  0.0f,0.0f,0.0f,
  1.0f,0.0f,0.0f,
  1.0f,1.0f,0.0f,
  0.0f,1.0f,0.0f,
  0.0f,0.0f,1.0f,
  1.0f,0.0f,1.0f,
  1.0f,1.0f,1.0f,
  0.0f,1.0f,1.0f
};
const static int numBoxFaces = 6;
const static int boxFaces[] = {
  0,1,2, 2,3,0,
  4,5,6, 6,7,4,
  0,1,5, 5,4,0,
  1,2,6, 6,5,1,
  2,3,7, 7,6,2,
  0,3,7, 7,4,0
};


#ifdef _WIN32
   #define isnan(x) ((x) != (x))
#endif

#define  SQR(x)  ((x)*(x))

#define  DEF_MIN(x,y)  ((x)<(y)?(x):(y))
#define  DEF_MAX(x,y)  ((x)>(y)?(x):(y))

#ifdef USE_DOUBLE
   typedef  double       real;
   #define  GL_REAL      GL_DOUBLE
   #define  REAL_MAX     1.79e308
   typedef  glm::dvec2   rvec2;
   typedef  glm::dvec3   rvec3;
   typedef  glm::dvec4   rvec4;
#else
   typedef  float        real;
   #define  GL_REAL      GL_FLOAT
   #define  REAL_MAX     3.4e38
   typedef  glm::vec2    rvec2;
   typedef  glm::vec3    rvec3;
   typedef  glm::vec4    rvec4;
#endif

const static int   numSimpleBoxVerts = 24;
const static float simpleBoxVerts[] = {
  0.0f,0.0f,0.0f,1.0f,  0.0f,0.0f,1.0f,1.0f,
  1.0f,0.0f,0.0f,1.0f,  1.0f,0.0f,1.0f,1.0f,
  1.0f,1.0f,0.0f,1.0f,  1.0f,1.0f,1.0f,1.0f,
  0.0f,1.0f,0.0f,1.0f,  0.0f,1.0f,1.0f,1.0f,
  0.0f,0.0f,0.0f,1.0f,  1.0f,0.0f,0.0f,1.0f,
  0.0f,1.0f,0.0f,1.0f,  1.0f,1.0f,0.0f,1.0f,
  0.0f,1.0f,1.0f,1.0f,  1.0f,1.0f,1.0f,1.0f,
  0.0f,0.0f,1.0f,1.0f,  1.0f,0.0f,1.0f,1.0f,
  0.0f,0.0f,0.0f,1.0f,  0.0f,1.0f,0.0f,1.0f,
  1.0f,0.0f,0.0f,1.0f,  1.0f,1.0f,0.0f,1.0f,
  1.0f,0.0f,1.0f,1.0f,  1.0f,1.0f,1.0f,1.0f,
  0.0f,0.0f,1.0f,1.0f,  0.0f,1.0f,1.0f,1.0f
};

/** Each object is identified by its object type.
 *   Take care that the enumeration is equivalent all over the code.
 *
 *   -> curvedSurfaceCode.cu
 *   -> GLShader.cpp  -> readShaderFromFile()
 *   -> curvedSurface.vert
 *   -> particleMapping.geom
 */
enum  e_surface_type
{
  e_surface_plane = 0,
  e_surface_sphere,
  e_surface_ellipsoid,
  e_surface_frustum,
  e_surface_torus,
  e_surface_moebius,
  e_surface_graph
};

/** Object structure for definition on host side.
 *    For particle simulations, some of the parameters have to be
 *    mapped to CSObjParams as declared in 'headers.cuh'.
 */
typedef struct Object_t
{
   e_surface_type  type;       //!<  object type
   
   int     slice_u;           //!<  number of slices in u
   int     slice_v;           //!<  number of slices in v
   
   rvec3   center;            //!<  center of object in global coordinates
   rvec3   e1;                //!<  local x-axis vector in global coordinates
   rvec3   e2;                //!<  local y-axis vector in global coordinates
   rvec3   e3;                //!<  local z-axis vector in global coordinates
   
   rvec2   u_range;           //!<  object is defined for u in u_range
   rvec2   v_range;           //!<  object is defined for v in v_range
   rvec2   uv_mod;            //!<  modulo values for particle motion
   glm::bvec2   use_modulo;   //!<  use modulo function for particle motion
   
   std::vector<double> params;  //!< object parameters
   bool    wireFrame;           //!< show object as wireframe

   // Default constructor
   Object_t() {
      type    = e_surface_plane;
      slice_u = 10;
      slice_v = 10;
      center  = rvec3(0,0,0);
      e1      = rvec3(1,0,0);
      e2      = rvec3(0,1,0);
      e3      = rvec3(0,0,1);
      u_range = rvec2(0.0,1.0);
      v_range = rvec2(0.0,1.0);
      uv_mod  = rvec2(1.0,1.0);
      use_modulo = glm::bvec2(false,false);
      wireFrame  = false;
   }
} Object;


typedef struct ParticleList_t
{
   unsigned int  obj_num;   //!< particle subset belongs to object number...
   unsigned int  num_i;     //!< initial particle number of subset
   unsigned int  num_f;     //!< final particle number of subset
} ParticleList;


/** Particle visualization type.
 */
enum  e_particleVisType
{
   e_particleVis_points_only = 0,    //!< show only points
   e_particleVis_splats_only,        //!< show only splats
   e_particleVis_points_and_splats   //!< show points and splats
};

#endif
