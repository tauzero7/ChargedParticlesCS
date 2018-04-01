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

/*  \file main.inl
 
      Add a new scene file by appending an 'elif' statement
      and an 'include'.
      Select a specific scene by changing 'USE_SCENE'.
*/


#define  USE_SCENE  0


// Background color
glm::vec3  bgColor = glm::vec3(0.1,0.1,0.1);
//glm::vec3  bgColor = glm::vec3(0.4,0.4,0.4);


// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


#if USE_SCENE   ==  0                  //!< Single sphere
#include <scenes/single_sphere.inl>

#elif USE_SCENE ==  1                  //!< Single torus
#include <scenes/single_torus.inl>

#elif USE_SCENE ==  2                  //!< Two spheres with particles of opposite charge
#include <scenes/two_spheres.inl>

#elif USE_SCENE ==  3                  //!< Two intertwined torii
#include <scenes/two_torii.inl>

#elif USE_SCENE ==  4                  //!> Single frustum
#include <scenes/single_frustum.inl>

#elif USE_SCENE ==  5                  //!< Moebius strip
#include <scenes/moebius.inl>

#elif USE_SCENE ==  6                  //!< Graph 
#include <scenes/graph.inl>

#elif USE_SCENE ==  7                  //!< Cube made of 6 planes
#include <scenes/cube.inl>

#elif USE_SCENE ==  8
#include <scenes/plane.inl>

#endif

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
