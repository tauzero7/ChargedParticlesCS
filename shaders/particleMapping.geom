

uniform mat4 proj_matrix;

uniform int    objType;
uniform float  ds;

uniform float param0;
uniform float param1;
uniform float param2;


layout(points) in;
layout(triangle_strip, max_vertices=4) out;

in  vec2 v_texCoords[];
in  vec4 v_color[];

out vec2 g_texCoords;
out vec4 g_color;

// ----------------------------------
//
// ----------------------------------
vec2  plane( in vec2 pos )
{  
   return vec2(ds,ds);
}

// ----------------------------------
//
// ----------------------------------
vec2  sphere( in vec2 pos )
{  
   float theta = pos.y;
   float r = param0;
   
   float dtheta = ds/r;
   float dphi   = ds/r/sin(theta);
   return vec2(dphi,dtheta);
}

// ----------------------------------
//
// ----------------------------------
vec2 ellipsoid( in vec2 pos )
{
   float a = param0;
   float b = param1;
   float c = param2;

   return vec2(ds,ds);  // TODO
}

// ----------------------------------
//
// ----------------------------------
vec2 frustum( in vec2 pos )
{
   float r1 = param0;
   float r2 = param1;
   float h  = param2;

   float rho = r1 - (r1-r2)/h*pos.y;
   float rhos = -(r1-r2)/h;

   float dphi = ds/rho;
   float dz   = ds/sqrt(rhos*rhos+1.0);
   return vec2(dphi,dz);
}

// ----------------------------------
//
// ----------------------------------
vec2  torus( in vec2 pos )
{
   float theta = pos.x;
   float Rdr = param0/param1;
   
   float dtheta = ds;
   float dphi   = ds/sqrt((Rdr + cos(theta))*(Rdr + cos(theta)));
   return vec2(dtheta,dphi);
}

// ----------------------------------
//
// ----------------------------------
vec2  moebius( in vec2 pos )
{  
   return vec2(ds,ds);
}

// ----------------------------------
//
// ----------------------------------
vec2  graph( in vec2 pos )
{
   float u = pos.x;
   float v = pos.y;
   
   float n = param0;
   float m = param1;
   float a = param2;
   
   float su = sin(2*PI*n*u);
   float cu = cos(2*PI*n*u);
   float sv = sin(2*PI*m*v);
   float cv = cos(2*PI*m*v);
   
   float a2 = a*a;
   float a4 = a2*a2;
   
   float g11 = sqrt(4*PI*PI*a4*n*n*cu*cu*sv*sv + 1.0);
   float g22 = sqrt(4*PI*PI*a4*m*m*su*su*cv*cv + 1.0);
   
   float du = ds/g11;
   float dv = ds/g22;
   return vec2(du,dv);
}

// ----------------------------------------------------------
//
// ----------------------------------------------------------
void main(void)
{
   float ts = 1.0;
   
   vec2 pos = gl_in[0].gl_Position.xy;   
   vec2 s   = vec2(ds,ds);
   
   switch (objType)
   {
      case SURF_TYPE_PLANE: {
         s = plane(pos);
         break;
      }
      case SURF_TYPE_SPHERE: {
         s = sphere(pos);
         break;
      }
      case SURF_TYPE_ELLIPSOID: {
         break;
      }
      case SURF_TYPE_FRUSTUM: {
         s = frustum(pos);
         break;
      }
      case SURF_TYPE_TORUS: {
         s = torus(pos);
         break;
      }
      case SURF_TYPE_MOEBIUS: {
         s = moebius(pos);
         break;
      }
      case SURF_TYPE_GRAPH: {
         s = graph(pos);
         break;
      }
   }
   
   gl_Position = proj_matrix * vec4(pos + vec2(-s.x,-s.y),0,1);
   g_texCoords = vec2(-ts,-ts);
   g_color = v_color[0];
   EmitVertex();
   
   gl_Position = proj_matrix * vec4(pos + vec2( s.x,-s.y),0,1);
   g_texCoords = vec2( ts,-ts);
   g_color = v_color[0];
   EmitVertex();
   
   gl_Position = proj_matrix * vec4(pos + vec2(-s.x, s.y),0,1);
   g_texCoords = vec2(-ts, ts);
   g_color = v_color[0];
   EmitVertex();
   
   gl_Position = proj_matrix * vec4(pos + vec2( s.x, s.y),0,1);
   g_texCoords = vec2( ts, ts);
   g_color = v_color[0];
   EmitVertex();
   
   EndPrimitive();
}
