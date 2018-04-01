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

//
//https://gitorious.org/wikibooks-opengl/modern-tutorials/blobs/master/text02_atlas/text.cpp
//

#include <RenderText.h>

/*! Standard constructor.
 * \param fontFilename : filename of font.
 * \param size : size of font.
 */
RenderText :: RenderText( std::string fontFilename, unsigned int size )
{
   fontFilename_ = fontFilename;
   fontSize_ = size;
   color_    = glm::vec3(1.0);
   coords_   = NULL;
   init();   
}

/*! Standard constructor.
 * \param fontFilename : filename of font.
 * \param size : size of font.
 * \param color : text color.
 */
RenderText :: RenderText( std::string fontFilename, unsigned int size, glm::vec3 color )
{
   fontFilename_ = fontFilename;
   fontSize_ = size;
   color_    = color;
   coords_   = NULL;
   init();     
}
   
/*! Standard destructor.
 */
RenderText :: ~RenderText()
{
   if (atlas_.texID>0)
      glDeleteTextures(1,&atlas_.texID);
      
   if (vbo_>0)
      glDeleteBuffers(1,&vbo_);
      
   if (va_>0)
      glDeleteVertexArrays(1,&va_);
}

/*! Set text color.
 * \param r : red value (0,1).
 * \param g : green value (0,1).
 * \param b : blue value (0,1).
 */
void  
RenderText :: setColor( float r, float g, float b ) {
   color_ = glm::vec3(r,g,b);   
}

/*! Set text color.
 * \param col : color (red,green,blue).
 */
void  
RenderText :: setColor( glm::vec3 col ) {
   color_ = col;
}

/*! Get text color.
 * \param r : reference to red value.
 * \param g : reference to green value.
 * \param b : reference to blue value.
 */
void  
RenderText :: getColor( float &r, float &g, float &b ) {
   r = color_[0];
   g = color_[1];
   b = color_[2];
}

/*! Render text.
 * \param text : text to render.
 * \param x : text position relative to the lower left corner (in pixel)
 * \param y : text position relative to the lower left corner (in pixel)
 * \param sx : text scale in x-direction (should be always 1)
 * \param sy : text scale in y-direction (should be always 1)
 */
void  
RenderText :: render( std::string text, float x, float y, float sx, float sy )
{
   const uint8_t *p;
   
   float winWidth  = (float)glutGet(GLUT_WINDOW_WIDTH);
   float winHeight = (float)glutGet(GLUT_WINDOW_HEIGHT);   
   //std::cerr << winWidth << " " << winHeight << std::endl;
   
   glEnable(GL_BLEND);
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   
   projMX_ = glm::ortho( 0.0f, winWidth, 0.0f, winHeight,-1.0f, 1.0f );
   
   shader.bind();
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, atlas_.texID);
 
   glUniform1i( unifLoc_tex_, 0 );
   glUniform3fv( unifLoc_color_, 1, &color_[0] );
   glUniformMatrix4fv( unifLoc_projMX_, 1, GL_FALSE, glm::value_ptr(projMX_) );
   
   glBindVertexArray(va_);     
   glBindBuffer( GL_ARRAY_BUFFER, vbo_ );
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0,4,GL_FLOAT,GL_FALSE,0,0);
   
   if (coords_!=NULL)
      delete [] coords_;
   coords_ = new glm::vec4[6*text.size()];
   
   int c = 0;   
   const char* t = text.c_str();
   for(p = (const uint8_t*)t; *p; p++)
   {
      float x2 =  x + atlas_.c[*p].bitmap_left * sx;
      float y2 = -y - atlas_.c[*p].bitmap_top  * sy;
      float w  = atlas_.c[*p].bitmap_width * sx;
      float h  = atlas_.c[*p].bitmap_height * sy;
      
      x += atlas_.c[*p].adv_x * sx;
      y += atlas_.c[*p].adv_y * sy;
      
      if (!w || !h)
         continue;
         
      coords_[c++] = glm::vec4( x2,     -y2,     atlas_.c[*p].tx,  0 );
      coords_[c++] = glm::vec4( x2 + w, -y2,     atlas_.c[*p].tx + atlas_.c[*p].bitmap_width / atlas_.width,  0 );
      coords_[c++] = glm::vec4( x2,     -y2 - h, atlas_.c[*p].tx,                                             atlas_.c[*p].bitmap_height/atlas_.height );
      coords_[c++] = glm::vec4( x2 + w, -y2,     atlas_.c[*p].tx + atlas_.c[*p].bitmap_width / atlas_.width,  0 );
      coords_[c++] = glm::vec4( x2,     -y2 - h, atlas_.c[*p].tx ,                                            atlas_.c[*p].bitmap_height/atlas_.height );
      coords_[c++] = glm::vec4( x2 + w, -y2 - h, atlas_.c[*p].tx + atlas_.c[*p].bitmap_width / atlas_.width,  atlas_.c[*p].bitmap_height/atlas_.height );      
   }
   
   glBufferData( GL_ARRAY_BUFFER, sizeof(glm::vec4)*6*text.size(), coords_, GL_DYNAMIC_DRAW);   
   glDrawArrays(GL_TRIANGLES, 0, c );
   
   glDisableVertexAttribArray(0);
   glBindBuffer( GL_ARRAY_BUFFER, 0 );
   glBindTexture(GL_TEXTURE_2D, 0);
   
   shader.release();
   glDisable(GL_BLEND);
}


void
RenderText :: printf( float x, float y, const char* fmt, ... )
{
   va_list ap;
   char text[256];
   if (fmt == NULL)
       return;

   va_start(ap, fmt);
   vsprintf(text, fmt, ap);
   va_end(ap);
   render(text,x,y);
}


/*! Initialize texture atlas and shaders.
 */
void  
RenderText :: init()
{
   fprintf(stderr,"Initialize RenderText...\n");
   if (FT_Init_FreeType(&ft_)) {
      fprintf(stderr,"@ %s (%d): Could not init freetype library.\n",__FILE__,__LINE__);
      return;
   }
   
   if (FT_New_Face(ft_, fontFilename_.c_str(), 0, &face_)) {
      char msg[256];
      sprintf(msg,"Could not open font %s.",fontFilename_.c_str());
      fprintf(stderr,"@ %s (%d): %s\n",__FILE__,__LINE__,msg);
      return;
   }
   
   FT_Set_Pixel_Sizes( face_, 0, fontSize_ );
   g_ = face_->glyph;
   
   int minw = 0;
   int minh = 0;
   memset(atlas_.c, 0, sizeof(atlas_.c));
   
   for(int i=32; i<128; i++) {
      if (FT_Load_Char(face_, i, FT_LOAD_RENDER)) {
         char msg[256];
         sprintf(msg,"Loading character %c failed.",i);
         fprintf(stderr,"@ %s (%d): %s\n",__FILE__,__LINE__,msg);
         continue;
      }
      minw += g_->bitmap.width + 1;
      minh  = std::max(minh, (int)g_->bitmap.rows);      
   }
   
   atlas_.width  = (float)minw;
   atlas_.height = (float)minh;
      
   glGenTextures(1, &atlas_.texID);
   
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D,atlas_.texID);    
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, minw, minh, 0, GL_RED, GL_UNSIGNED_BYTE, 0);
   glPixelStorei(GL_UNPACK_ALIGNMENT,1);
   
   int o=0;
   for(int i=32; i<128; i++) {
      if (FT_Load_Char(face_, i, FT_LOAD_RENDER)) {
         char msg[256];
         sprintf(msg,"Loading character %c failed.",i);
         fprintf(stderr,"@ %s (%d): %s\n",__FILE__,__LINE__,msg);
         continue;
      }
      glTexSubImage2D(GL_TEXTURE_2D, 0, o, 0, g_->bitmap.width, g_->bitmap.rows, GL_RED, GL_UNSIGNED_BYTE, g_->bitmap.buffer);
      
      atlas_.c[i].adv_x = g_->advance.x >> 6;
      atlas_.c[i].adv_y = g_->advance.y >> 6;
      atlas_.c[i].bitmap_width  = g_->bitmap.width;
      atlas_.c[i].bitmap_height = g_->bitmap.rows;
      atlas_.c[i].bitmap_left   = g_->bitmap_left;
      atlas_.c[i].bitmap_top    = g_->bitmap_top;
      atlas_.c[i].tx = o/atlas_.width;
      o += g_->bitmap.width + 1;
   }   
   glGenerateMipmap(GL_TEXTURE_2D);   
   fprintf(stderr,"Generated a %d x %d (%d kb) texture atlas\n", minw, minh, minw * minh / 1024);
   
   if (!shader.createProgramFromString(vDTShaderText,fDTShaderText)) {
      fprintf(stderr,"@ %s (%d): Cannot create shader for drawing text.\n",__FILE__,__LINE__);
   }
   
   shader.bind();
   unifLoc_projMX_ = glGetUniformLocation( shader.prog(), "projMX" );
   unifLoc_tex_    = glGetUniformLocation( shader.prog(), "tex" );
   unifLoc_color_  = glGetUniformLocation( shader.prog(), "color" );
   //std::cerr << unifLoc_projMX_ << " " << unifLoc_tex_ << " " << unifLoc_color_ << std::endl;
   shader.release();
   
   glGenVertexArrays(1,&va_);
   glGenBuffers(1,&vbo_);
}
