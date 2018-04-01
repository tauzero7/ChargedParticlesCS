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

#include "utils.h"

/**  Get system clock in micro-seconds.
 *  \return  int64_t : system clock in microseconds.
 */
#ifndef _WIN32
int64_t  get_system_clock() 
{
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return (int64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}
#endif


/**  Check for standard OpenGL errors.
 *   \file : filename
 *   \line : line number
 */
GLenum checkError(std::string file, int line) {
    GLenum __theError;
    __theError = glGetError();
    if (__theError != GL_NO_ERROR) {
        switch(__theError) {
            case GL_INVALID_ENUM:
                printf("GL_INVALID_ENUM at %s:%u\n", file.c_str(), line);
                break;
            case GL_INVALID_VALUE:
                printf("GL_INVALID_VALUE at %s:%u\n", file.c_str(), line);
                break;
            case GL_INVALID_OPERATION:
                printf("GL_INVALID_OPERATION at %s:%u\n", file.c_str(), line);
                break;
            case GL_OUT_OF_MEMORY:
                printf("GL_OUT_OF_MEMORY at %s:%u\n", file.c_str(), line);
                break;
        }
    }
    return __theError;
}


/** Print cuda device properties
 * \param deviceProp : pointer to device properties.
 * \param fptr : file pointer.
 */
void  printDeviceProp ( cudaDeviceProp* deviceProp , FILE* fptr )
{
   fprintf(fptr,"\ttotalGlobalMem     : %d\n",(int)deviceProp->totalGlobalMem);
   fprintf(fptr,"\tsharedMemPerBlock  : %d\n",(int)deviceProp->sharedMemPerBlock);
   fprintf(fptr,"\tregsPerBlock       : %d\n",deviceProp->regsPerBlock);
   fprintf(fptr,"\twarpSize           : %d\n",deviceProp->warpSize);
   fprintf(fptr,"\tmemPitch           : %d\n",(int)deviceProp->memPitch);
   fprintf(fptr,"\tmaxThreadsPerBlock : %d\n",deviceProp->maxThreadsPerBlock);
   fprintf(fptr,"\tmaxThreadsDim      : %d %d %d\n",deviceProp->maxThreadsDim[0],deviceProp->maxThreadsDim[1],deviceProp->maxThreadsDim[2]);
   fprintf(fptr,"\tmaxGridSize        : %d %d %d\n",deviceProp->maxGridSize[0],deviceProp->maxGridSize[1],deviceProp->maxGridSize[2]);
   fprintf(fptr,"\tclockRate          : %d\n",deviceProp->clockRate);
   fprintf(fptr,"\ttotalConstMem      : %d\n",(int)deviceProp->totalConstMem);
   fprintf(fptr,"\tcomput. cap. major : %d\n",deviceProp->major);
   fprintf(fptr,"\tcomput. cap. minor : %d\n",deviceProp->minor);
   fprintf(fptr,"\ttextureAlignment   : %d\n",(int)deviceProp->textureAlignment);
   fprintf(fptr,"\tdeviceOverlap      : %d\n",deviceProp->deviceOverlap);
   fprintf(fptr,"\tmultiProcCount     : %d\n",deviceProp->multiProcessorCount);
   fprintf(fptr,"\n");
}


/** Print  vec3 with real entries
 * \param v : vector
 */
void print_vec3( rvec3 v)
{
   printf("vec3(%7.3f, %7.3f, %7.3f)\n",v[0],v[1],v[2]);
}

void printf_vec3( rvec3 v, FILE* fptr )
{
   fprintf(fptr,"%12.6f, %12.6f, %12.6f\n",v[0],v[1],v[2]);
}

/** Print mat4 matrix
 * \param m : 4x4-matrix
 */
void print_mat4(glm::mat4 m)
{
    printf("mat4(\n");
    printf("\tvec4(%7.3f, %7.3f, %7.3f, %7.3f)\n", m[0][0], m[0][1], m[0][2], m[0][3]);
    printf("\tvec4(%7.3f, %7.3f, %7.3f, %7.3f)\n", m[1][0], m[1][1], m[1][2], m[1][3]);
    printf("\tvec4(%7.3f, %7.3f, %7.3f, %7.3f)\n", m[2][0], m[2][1], m[2][2], m[2][3]);
    printf("\tvec4(%7.3f, %7.3f, %7.3f, %7.3f))\n\n", m[3][0], m[3][1], m[3][2], m[3][3]);
}

/** Get OpenGL information from system.
 * 
 */
void   getOpenGLInfo( FILE* fptr )
{
  const GLubyte *ven = glGetString(GL_VENDOR);
  ven = glGetString(GL_VERSION);
  
//  GLint maxVertexAttribs;
//  glGetIntegerv(GL_MAX_VERTEX_ATTRIBS,&maxVertexAttribs);
  
//  fprintf(fptr,"OpenGLInfo:\n-----------\n");
//  fprintf(fptr,"GL_MAX_VERTEX_ATTRIBS : %d\n",maxVertexAttribs);
}

/** Get random value
 */
double getRandomValue()
{
   return  (rand() % RAND_MAX)/(double)RAND_MAX;
}
