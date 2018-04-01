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

#ifndef GL_SHADER_H
#define GL_SHADER_H

#include <iostream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>

#include <defs.h>
#include <utils.h>

#ifdef _WIN32
#else 
#include <GL/glx.h>
#endif // _WIN32

class GLShader
{
 public:
   GLShader();
   ~GLShader();

   bool    createProgram ( const std::string vShaderName, const std::string fShaderName ); 
   
   bool    createProgram ( const std::string vShaderName, const std::string gShaderName, const std::string fShaderName,
                           const GLenum inputType, const GLenum outputType, const int numOutputVertices ); 
   
   bool    createProgramFromString( const std::string vShaderText, const std::string fShaderText );
   
   void    bind             ( );
   void    release          ( );
   GLuint  prog             ( );
   GLint   uniformLocation  ( std::string name );
   
   void    removeAllShaders ( );
   bool    reload           ( );
   void    printInfo        ( FILE* fptr = stderr );
   
   void    setUniformValue  ( std::string name, int val );
   void    setUniformValue  ( std::string name, int val1, int val2 );
   void    setUniformValue  ( std::string name, float val );
   void    setUniformValue  ( std::string name, float val1, float val2 );
   void    setUniformValue  ( std::string name, float val1, float val2, float val3 );
   void    setUniformValue  ( std::string name, float val1, float val2, float val3, float val4 );

   
// ---------- protected methods -----------
protected:   
   GLuint  readShaderFromFile     ( std::string shaderFilename, std::string &shaderContent );
   GLuint  createShaderFromFile   ( std::string shaderFilename, GLenum type );
   
   GLuint  createShaderFromString ( std::string shaderText, GLenum type );
   
   bool    printShaderInfoLog     ( GLuint shader );
   bool    printProgramInfoLog    ();   

// ---------- private attributes --------
 private:
   GLuint       m_glProg;
   bool         m_withGeomShader;
   
   std::string  m_vertShaderName;
   std::string  m_geomShaderName;
   std::string  m_fragShaderName;
   
   GLenum       m_inputType;
   GLenum       m_outputType;
   int          m_numOutputVertices;
};
  
#endif  // GL_SHADER_H
