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

#ifndef RENDER_TEXT_H
#define RENDER_TEXT_H

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#include <defs.h>
#include <GLShader.h>
#include <GL/freeglut.h>

#include <ft2build.h>
#include FT_FREETYPE_H
   
typedef struct char_info_t
{
    float adv_x;         //!< advance in x direction
    float adv_y;         //!< advance in y direction
    float bitmap_width;  
    float bitmap_height;
    float bitmap_left;
    float bitmap_top;
    float tx;            //!< x offset of glyph in texture coordinates
} char_info;
   
   
typedef struct font_atlas_t
{
   GLuint texID;        //!< texture id
   float  width;        //!< width of texture
   float  height;       //!< height of texture
   char_info  c[128];   //!< character information
} font_atlas;   
   
   
const std::string  vDTShaderText = std::string("#version 330\n\n"  
   "uniform mat4 projMX;\n"
   "layout( location = 0) in  vec4 coord;\n"
   "out vec2 texpos;\n"
   "void main(void) {\n"
   "  gl_Position = projMX * vec4(coord.xy, 0, 1);\n"
   "  texpos = coord.zw;\n"
   "}\n");
   
const std::string  fDTShaderText = std::string("#version 330\n\n"
   "uniform sampler2D tex;\n"
   "uniform vec3 color;\n"
   "in  vec2 texpos;\n"
   "layout( location = 0 ) out vec4 out_frag_color;\n"
   "void main(void) {\n"
   "  out_frag_color = vec4(1, 1, 1, texture2D(tex, texpos).r) * vec4(color,1.0);\n"
   //"  out_frag_color = vec4(1,1,0,1);\n"
   "}\n");
   
   
   
class RenderText
{
 public:    
   RenderText( std::string fontFilename, unsigned int size );
   RenderText( std::string fontFilename, unsigned int size, glm::vec3 color );
   ~RenderText();
   
   void  setColor    ( float r, float g, float b );   
   void  setColor    ( glm::vec3 col );
   void  getColor    ( float &r, float &g, float &b );
   
   void  render      ( std::string text, float x, float y, float sx = 1.0, float sy = 1.0 );
   void  printf      ( float x, float y, const char* fmt, ... );
   
   
 protected:
   void  init ();   
   
 private:
   std::string  fontFilename_;
   unsigned int fontSize_;
   
   GLuint       va_;
   GLuint       vbo_;
   GLint        unifLoc_projMX_;
   GLint        unifLoc_tex_;
   GLint        unifLoc_color_;
   
   FT_Library   ft_;
   FT_Face      face_;
   FT_GlyphSlot g_;
   font_atlas   atlas_;
   
   GLShader     shader;
   glm::mat4    projMX_;
   glm::vec3    color_;
   glm::vec4*   coords_;
};

#endif // RENDER_TEXT_H
