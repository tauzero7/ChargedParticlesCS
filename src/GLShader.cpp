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

#include "GLShader.h"

/**
 */
GLShader :: GLShader()
{
   m_glProg = 0;
   m_withGeomShader = false;
}

/**
 */
GLShader :: ~GLShader()
{
   removeAllShaders();
}

/** Create program
 *  \param vShaderName : name of vertex shader file.
 *  \param fShaderName : name of fragment shader file.
 *  \return true : successfull creation.
 */
bool
GLShader :: createProgram ( const std::string vShaderName, const std::string fShaderName )
{
   fprintf(stderr,"Create Program\n\tVertex shader:   %s\n\tFragment shader: %s\n",vShaderName.c_str(),fShaderName.c_str());
   GLuint vShader = createShaderFromFile(vShaderName,GL_VERTEX_SHADER);
   if (vShader==0)  return false;
   
   GLuint fShader = createShaderFromFile(fShaderName,GL_FRAGMENT_SHADER);
   if (fShader==0)  return false;

   m_vertShaderName = vShaderName;
   m_fragShaderName = fShaderName;
   
   m_glProg = glCreateProgram();
   glAttachShader( m_glProg, vShader );
   glAttachShader( m_glProg, fShader );

   glLinkProgram(m_glProg);
   bool status = printProgramInfoLog();
   glUseProgram(0);
   return (status==GL_TRUE);
}

/*! Create program
 *  \param vShaderName : name of vertex shader file.
 *  \param gShaderName : name of geometry shader file.
 *  \param fShaderName : name of fragment shader file.
 *  \param inputType : input type for geometry shader.
 *  \param outputType : output type for geometry shader.
 *  \param numOutputVertices : number of output vertices from geometry shader.
 *  \return true : successfull creation.
 */
bool    
GLShader :: createProgram ( const std::string vShaderName, const std::string gShaderName, const std::string fShaderName,
                            const GLenum inputType, const GLenum outputType, const int numOutputVertices )
{
   fprintf(stderr,"Create Program\n\tVertex shader:   %s\n\tGeometry shader: %s\n\tFragment shader: %s\n",
            vShaderName.c_str(),fShaderName.c_str(),fShaderName.c_str());
   GLuint vShader = createShaderFromFile(vShaderName,GL_VERTEX_SHADER);
   if (vShader==0)  return false;
   
   GLuint gShader = createShaderFromFile(gShaderName,GL_GEOMETRY_SHADER);
   if (gShader==0)  return false;
   m_withGeomShader = true;
   
   GLuint fShader = createShaderFromFile(fShaderName,GL_FRAGMENT_SHADER);
   if (fShader==0)  return false;

   m_vertShaderName = vShaderName;
   m_geomShaderName = gShaderName;
   m_fragShaderName = fShaderName;
   
   m_glProg = glCreateProgram();
   glAttachShader( m_glProg, vShader );
   glAttachShader( m_glProg, gShader );
   glAttachShader( m_glProg, fShader );

   glProgramParameteri( m_glProg, GL_GEOMETRY_INPUT_TYPE,   inputType );
   glProgramParameteri( m_glProg, GL_GEOMETRY_OUTPUT_TYPE,  outputType );
   glProgramParameteri( m_glProg, GL_GEOMETRY_VERTICES_OUT, numOutputVertices );
   m_inputType         = inputType;
   m_outputType        = outputType;
   m_numOutputVertices = numOutputVertices;

   glLinkProgram(m_glProg);
   bool status = printProgramInfoLog();
   glUseProgram(0);
   return (status==GL_TRUE);
}

/**
 */
bool    
GLShader :: createProgramFromString( const std::string vShaderText, const std::string fShaderText )
{
   GLuint vShader = createShaderFromString(vShaderText,GL_VERTEX_SHADER);
   if (vShader==0)  return false;
   GLuint fShader = createShaderFromString(fShaderText,GL_FRAGMENT_SHADER);
   if (fShader==0)  return false;
   
   m_glProg = glCreateProgram();
   glAttachShader( m_glProg, vShader );
   glAttachShader( m_glProg, fShader );
   
   glLinkProgram(m_glProg);
   bool status = printProgramInfoLog();
   glUseProgram(0);
   return (status==GL_TRUE);
}

/**
 */
void
GLShader :: bind() {
   glUseProgram(m_glProg);
}

/**
 */
void
GLShader :: release() {
   glUseProgram(0);
}

/**
 */
GLuint  
GLShader :: prog() {
   return m_glProg;
}

/**
 */
GLint
GLShader :: uniformLocation( std::string name ) {
   return glGetUniformLocation( m_glProg, name.c_str() );
}

/*! Read shader from file.
 * 
 */
GLuint
GLShader :: readShaderFromFile( std::string shaderFilename, std::string &shaderContent )
{
   std::ifstream in(shaderFilename.c_str());  
   if (!in) {
      char msg[256];
      sprintf(msg,"Cannot open file \"%s\"\n",shaderFilename.c_str());
      fprintf(stderr,"@ %s (%d): %s\n",__FILE__,__LINE__,msg);
      return 0;
   }

   std::stringstream shaderData;
   
   shaderData << "#version 330\n";
   shaderData << "#define PI     3.14159265\n";
   shaderData << "#define PI_2   1.5707963\n";
   shaderData << "#define invPI  0.31830989\n";
   
   shaderData << "#define   SURF_TYPE_PLANE         0\n";
   shaderData << "#define   SURF_TYPE_SPHERE        1\n";
   shaderData << "#define   SURF_TYPE_ELLIPSOID     2\n";
   shaderData << "#define   SURF_TYPE_FRUSTUM       3\n";
   shaderData << "#define   SURF_TYPE_TORUS         4\n";
   shaderData << "#define   SURF_TYPE_MOEBIUS       5\n";
   shaderData << "#define   SURF_TYPE_GRAPH         6\n";
   
   shaderData << in.rdbuf();
   in.close();
   
   shaderContent = shaderData.str();
   return shaderContent.size();
}


/*! Compile shader from file.
 * \param shaderText : shader text.
 * \param type : shader type.
 */
 GLuint 
 GLShader :: createShaderFromFile( std::string shaderFilename, GLenum type )
{
   std::string shaderText;
   GLuint iShaderLen = readShaderFromFile(shaderFilename,shaderText);
   if (iShaderLen==0)
      return 0;

   GLuint shader = glCreateShader(type);
   const char *strShaderVar = shaderText.c_str();
   glShaderSource(shader,1,(const GLchar**)&strShaderVar, (GLint*)&iShaderLen);
   glCompileShader(shader);
    
   if (!printShaderInfoLog(shader)) {
      std::stringstream iss(shaderText);
      std::string sLine;
      int lineCounter = 1;
      while(std::getline(iss,sLine)) {
         fprintf(stderr,"%4d : %s\n",(lineCounter++),sLine.c_str());
      }
   }
   return shader;
}

/*! Create shader from string.
 * \param shaderText : shader text.
 * \param type : shader type.
 * \return shader id.
 */
GLuint  
GLShader :: createShaderFromString ( std::string shaderText, GLenum type )
{
   GLuint shader = glCreateShader(type);
   if (shader>0) {
      const GLchar* source = shaderText.c_str();
      GLCE(glShaderSource(shader, 1, &source, NULL));
      GLCE(glCompileShader(shader));
      
      if (!printShaderInfoLog(shader)) {
         std::stringstream iss(shaderText);
         std::string sLine;
         int lineCounter = 1;
         while(std::getline(iss,sLine)) {
            fprintf(stderr,"%4d : %s\n",(lineCounter++),sLine.c_str());
         }
      }
   }
   return shader;
}

/*! Print shader information log.
 * 
 */
bool
GLShader :: printShaderInfoLog( GLuint shader ) 
{
   int infoLogLen   = 0;
   int charsWritten = 0;
   GLchar *infoLog;

   int compileStatus = 0;
   glGetShaderiv(shader, GL_COMPILE_STATUS,  &compileStatus);
   glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLen);

   if (infoLogLen > 1)  {
      infoLog = new GLchar[infoLogLen];
      // error check for fail to allocate memory omitted
      glGetShaderInfoLog(shader, infoLogLen, &charsWritten, infoLog);
      printf("InfoLog : %s\n", infoLog);
      delete [] infoLog;
   }
   return (compileStatus == GL_TRUE);
}

/*! Print program information log.
 * 
 */
bool 
GLShader :: printProgramInfoLog() 
{
   int infoLogLen   = 0;
   int charsWritten = 0;
   GLchar *infoLog;

   int linkStatus = 0;
   glGetProgramiv(m_glProg, GL_INFO_LOG_LENGTH, &infoLogLen);
   glGetProgramiv(m_glProg, GL_LINK_STATUS, &linkStatus);

   if (infoLogLen > 1)  {
      infoLog = new GLchar[infoLogLen];
      // error check for fail to allocate memory omitted
      glGetProgramInfoLog(m_glProg, infoLogLen, &charsWritten, infoLog);
      printf("ProgramInfoLog : %s\n", infoLog);
      delete [] infoLog;
   }
   return (linkStatus == GL_TRUE);
}

/**
 */
void  
GLShader :: removeAllShaders()
{
   if (!glIsProgram(m_glProg)) {
      return;
   }
   
   const GLsizei numShaders = 1024;
   GLsizei numReturned;
   GLuint shaders[numShaders];
   GLCE(glUseProgram(0));
   
   GLCE(glGetAttachedShaders(m_glProg, numShaders, &numReturned, shaders));
   for (GLsizei i = 0; i < numReturned; i++) {
      GLCE(glDetachShader(m_glProg, shaders[i]));
      GLCE(glDeleteShader(shaders[i]));
   }
   glDeleteProgram(m_glProg);
   m_glProg = 0;
}

/**
 */
bool
GLShader :: reload() 
{
   removeAllShaders();
   if (!m_withGeomShader)
      return createProgram(m_vertShaderName,m_fragShaderName);
   else
      return createProgram(m_vertShaderName,m_geomShaderName,m_fragShaderName,
                           m_inputType,m_outputType,m_numOutputVertices);
   
}

/**
 */
void
GLShader :: setUniformValue( std::string name, int val ) {
   glUniform1i( uniformLocation(name), val );
}

/**
 */
void    
GLShader :: setUniformValue( std::string name, int val1, int val2 ) {
   glUniform2i( uniformLocation(name), val1, val2 );
}

/**
 */
void
GLShader :: setUniformValue( std::string name, float val ) {
   glUniform1f( uniformLocation(name), val );
}

/**
 */
void
GLShader :: setUniformValue( std::string name, float val1, float val2 ) {
   glUniform2f( uniformLocation(name), val1, val2 );
}

/**
 */
void
GLShader :: setUniformValue( std::string name, float val1, float val2, float val3 ) {
   glUniform3f( uniformLocation(name), val1, val2, val3 );
}

/**
 */
void
GLShader :: setUniformValue( std::string name, float val1, float val2, float val3, float val4 ) {
   glUniform4f( uniformLocation(name), val1, val2, val3, val4 );
}
