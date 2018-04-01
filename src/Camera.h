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

#ifndef CAMERA_H
#define CAMERA_H

#include <iostream>
#include <cstdio>

extern "C" {
#include <GL3/gl3w.h>
}

#include <defs.h>

class Camera
{
 public: 
  Camera ( ); 
  virtual ~Camera ( );
 
// --------- public methods -----------
 public:
   void   setStandardParams  ( );
 
   void        set        ( glm::vec3 pos, glm::vec3 dir, glm::vec3 vup );
   void        setPosPOI  ( glm::vec3 pos, glm::vec3 poi, glm::vec3 vup );
 
   void        setEyePos  ( float eye_x, float eye_y, float eye_z );
   void        setEyePos  ( glm::vec3 pos );
   void        getEyePos  ( float &eye_x, float &eye_y, float &eye_z );     
   glm::vec3   getEyePos  ( );
 
   void        setDir     ( float dir_x, float dir_y, float dir_z );
   void        setDir     ( glm::vec3 dir );
   void        getDir     ( float &dir_x, float &dir_y, float &dir_z );
   glm::vec3   getDir     ( );
   
   glm::vec3   pixelToDir ( int i, int j );

   void        setPOI     ( float c_x, float c_y, float c_z );
   void        setPOI     ( glm::vec3 center );
   void        getPOI     ( float &c_x, float &c_y, float &c_z );
   glm::vec3   getPOI     ( );
   
   void        setVup     ( float vup_x, float vup_y, float vup_z );
   void        setVup     ( glm::vec3 vup );
   void        getVup     ( float &vup_x, float &vup_y, float &vup_z );
   glm::vec3   getVup     ( );
   
   glm::vec3   getRight   ( );
   
   void        setFovY      ( float fovy );   
   float       getFovY      ( );
  
   void        setAspect    ( float aspect );   
   float       getAspect    ( );
    
   void        setIntrinsic ( float fovy, float aspect );   
   void        setIntrinsic ( float fovy, float aspect, float near, float far );   
   void        getIntrinsic ( float &fovy, float &aspect, float &near, float &far );
   void        setNearFar   ( float near, float far );
   void        getNearFar   ( float &near, float &far );
       
   void        setSizeAndAspect  ( int width, int  height );
 
   void        setSize           ( int width, int height );
   void        getSize           ( int &width, int &height );
   int         width             ( );
   int         height            ( );
  
   void        fixRotAroundVup   ( float angle );   
   void        fixRotAroundRight ( float angle );   
   void        fixRotAroundDir   ( float angle );

   void        rotAroundVup      ( float angle );  
   void        rotAroundRight    ( float angle );   
   void        rotAroundDir      ( float angle );
      
   void        moveOnSphere      ( float delta_theta, float delta_phi );
   void        changeDistance    ( float deltaDist );
   void        panning           ( float dx, float dy );
   void        moveOnXY          ( float dx, float dy );
   void        moveOnZ           ( float dz );

   void        setDistance       ( float distance );
   float       getDistance       ();
   void        setTheta          ( float theta );
   float       getTheta          ();
   void        setPhi            ( float phi );
   float       getPhi            ();

   void        setMinDistance    ( float min_dist );
     
   void        setClearColors ( glm::vec4 col );
   glm::vec4   getClearColors ( );
   
   void        setViewMX_lookAt  ( bool upsideDown = false ); 
   void        setProjMX_persp   ( );   
   void        viewport          ( );
   
   float*      projMatrixPtr  ( );
   float*      viewMatrixPtr  ( );
   
   glm::mat4   projMatrix     ( );
   glm::mat4   viewMatrix     ( );
      
   void        setPanStep     ( float panStep );
   float       getPanStep     ( );
   void        setDistFactor  ( float distFactor );
   float       getDistFactor  ( );
   void        setPanXYFactor ( float panXYfactor );
   float       getPanXYFactor ( );
   void        setRotFactor   ( float rotFactor );
   float       getRotFactor   ( );
      
   void        print ( FILE* ptr = stderr );
   
// --------- protected attributes -----------
 protected:
   glm::vec3  m_pos;
   glm::vec3  m_dir;
   glm::vec3  m_vup;
   
   GLfloat    m_zNear;
   GLfloat    m_zFar;
   GLfloat    m_aspect;
   GLint      m_width;
   GLint      m_height;
   GLfloat    m_fovY;
 
   glm::vec3  m_right;
   glm::vec3  m_poi;
 
   GLfloat    m_dist;
   GLfloat    m_min_dist;
   
   glm::vec4  m_clearColor;
   
   glm::mat4  m_projMX;
   glm::mat4  m_viewMX;
   
   float      m_panStep;
   float      m_distFactor;
   float      m_panXYfactor;
   float      m_rotFactor;
};

#endif  // CAMERA_H
