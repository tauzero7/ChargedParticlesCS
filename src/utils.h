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

#ifndef CL_UTILS_H
#define CL_UTILS_H

#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <sstream>
#include <cstdlib>

#include <defs.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#ifndef _WIN32
#include <ctime>
#include <sys/time.h>
int64_t  get_system_clock ( );
#endif


#define GLCE(x) glGetError(); (x); checkError(__FILE__, __LINE__);

#define CUDA_SAFE_CALL_NO_SYNC(call) {                                    \
   cudaError err = call;                                                  \
   if( cudaSuccess != err) {                                              \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",       \
              __FILE__, __LINE__, cudaGetErrorString( err) );             \
      exit(EXIT_FAILURE);                                                 \
   } }

#define CUDA_SAFE_CALL(call)     CUDA_SAFE_CALL_NO_SYNC(call);   


#define CUDA_CHECK_ERROR() {                                              \
   cudaError err = cudaGetLastError();                                    \
   if ( cudaSuccess != err )  {                                           \
      fprintf( stderr, "Cuda error in file '%s' in line %i : %s\n",       \
               __FILE__,__LINE__, cudaGetErrorString( err ) );            \
        exit(EXIT_FAILURE);                                               \
    } }


GLenum  checkError(std::string file, int line);

void    printDeviceProp ( cudaDeviceProp* deviceProp , FILE* fptr );
void    print_vec3  ( rvec3 v);
void    printf_vec3 ( rvec3 v, FILE* fptr = stderr );
void    print_mat4  ( glm::mat4 m );

void    getOpenGLInfo  ( FILE* fptr = stderr );
double  getRandomValue();

#endif
