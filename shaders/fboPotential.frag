
uniform vec3  particle_pos;
uniform float particle_q;

uniform int   objType;
uniform vec2  uv_min;
uniform vec2  uv_max;

uniform vec3  center;
uniform vec3  e1;
uniform vec3  e2;
uniform vec3  e3;

uniform float param0;
uniform float param1;
uniform float param2;


in vec2 texCoords;

layout(location = 0) out vec4 out_frag_color;


// ----------------------------------------------------------
//   u : phi,  v : theta
// ----------------------------------------------------------
void sphere( in float u, in float v, out vec3 pos )
{
   float r = param0;
   float x = r*sin(v)*cos(u);
   float y = r*sin(v)*sin(u);
   float z = r*cos(v);   
   
   pos = x*e1 + y*e2 + z*e3 + center;
}

// ----------------------------------------------------------
//
// ----------------------------------------------------------
void torus( in float u, in float v, out vec3 pos)
{
   float R = param0;
   float r = param1;
   float x = (R + r*cos(u)) * cos(v);
   float y = (R + r*cos(u)) * sin(v);
   float z = r*sin(u);   
   
   pos  = x*e1 + y*e2 + z*e3 + center;
}


// ----------------------------------------------------------
//
// ----------------------------------------------------------
void main(void)
{
   vec2 uv = (uv_max - uv_min)*texCoords + uv_min;

   vec3 pos = vec3(0);
   switch (objType)
   {
      case SURF_TYPE_SPHERE: {
         sphere(uv.x,uv.y,pos);
         break;
      } 
      case SURF_TYPE_TORUS: {
         torus(uv.x,uv.y,pos);
         break;
      }
   }
   
   float dist = length(pos-particle_pos);
   float val = particle_q/(dist);
   
   vec4 col = vec4(0.0);

#if 1
   if (particle_q<0) col = vec4(vec3(1.0,0.05,0.01)*val,abs(val));
   else              col = vec4(vec3(0.01,0.05,1.0)*val,abs(val));
#else  
   if (particle_q<0) col = vec4(vec3(1.0,1.0,1.0)*val,abs(val));
   else              col = vec4(vec3(1.0,1.0,1.0)*val,abs(val));
#endif
   col = vec4(1,1,0,1);
   out_frag_color = col;
}
