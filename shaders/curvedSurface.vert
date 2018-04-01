
layout(location = 0) in vec4 in_position;
layout(location = 1) in vec4 in_velocity;
layout(location = 2) in vec4 in_color;

uniform mat4 proj_matrix;
uniform mat4 view_matrix;

uniform int   isSurface;
uniform int   objType;
uniform vec2  uv_min;
uniform vec2  uv_max;
uniform vec2  uvStep;
uniform ivec2 slice;
uniform vec3  center;
uniform vec3  e1;
uniform vec3  e2;
uniform vec3  e3;
uniform int   chart;

uniform float param0;
uniform float param1;
uniform float param2;
uniform float param3;

out vec2 texCoords;
out vec3 localPos;
out vec3 normDir;
out vec4 color;

flat out int  doDiscardVertex;

// ----------------------------------------------------------
//   u : phi,  v : theta
// ----------------------------------------------------------
void plane( in float u, in float v, in float scale,
            out vec4 vx, out vec3 ndir, out vec2 tx )
{
   float x = u;
   float y = v;
   float z = 0;
   
   vx   = vec4(x,y,z,1);
   tx   = (vec2(u,v) - uv_min)/(uv_max - uv_min);
   
   ndir = vec3(0,0,1);
}

// ----------------------------------------------------------
//   u : phi,  v : theta
// ----------------------------------------------------------
void sphere( in float u, in float v, in float scale,
             out vec4 vx, out vec3 ndir, out vec2 tx )
{
   float r = param0*scale;
   float x = r*sin(v)*cos(u);
   float y = r*sin(v)*sin(u);
   float z = r*cos(v);   
   
   vx   = vec4(x,y,z,1);
   tx   = (vec2(u,v) - uv_min)/(uv_max - uv_min);
   
   ndir = normalize(vx.xyz);
}

// ----------------------------------------------------------
//   u : phi,  v : theta
// ----------------------------------------------------------
void ellipsoid( in float u, in float v, in float scale,
                out vec4 vx, out vec3 ndir, out vec2 tx )
{
   float a = param0*scale;
   float b = param1*scale;
   float c = param2*scale;

   float phi = u;
   float theta = v;

   float x = a*cos(theta)*cos(phi);
   float y = b*cos(theta)*sin(phi);
   float z = c*sin(theta);
   
   vx   = vec4(x,y,z,1);
   tx   = (vec2(u,v) - uv_min)/(uv_max - uv_min);
   
   ndir = normalize(vx.xyz);  // ???
}

// ----------------------------------------------------------
//   u : phi,  v : z
// ----------------------------------------------------------
void frustum( in float u, in float v, in float scale,
              out vec4 vx, out vec3 ndir, out vec2 tx )
{
   float r1 = param0*scale;
   float r2 = param1*scale;
   float h  = param2*scale;

   float rho = r1 - (r1-r2)/h*v;
   float rhos = -(r1-r2)/h;

   float x = rho*cos(u);
   float y = rho*sin(u);
   float z = v;
   
   vx   = vec4(x,y,z,1);
   tx   = (vec2(u,v) - uv_min)/(uv_max - uv_min);
   
   ndir = vec3(cos(u),sin(u),-rhos)/sqrt(1+rhos*rhos);
}

// ----------------------------------------------------------
//
// ----------------------------------------------------------
void torus( in float u, in float v, in float scale,
            out vec4 vx, out vec3 ndir, out vec2 tx )
{
   float R = param0;
   float r = param1*scale;
   float x = (R + r*cos(u)) * cos(v);
   float y = (R + r*cos(u)) * sin(v);
   float z = r*sin(u);   
   
   vx   = vec4(x,y,z,1);
   tx   = vec2(u,v)*invPI*0.5;
   
   vec3 du = vec3(-r*sin(u)*cos(v),-r*sin(u)*sin(v),r*cos(u));
   vec3 dv = vec3(-(R+r*cos(u))*sin(v),(R+r*cos(u))*cos(v),0);
   
   ndir = normalize(-cross(du,dv));
}

// ----------------------------------------------------------
//
// ----------------------------------------------------------
/*
void torusKnot( in float u, in float v, in float scale,
                out vec4 vx, out vec3 ndir, out vec2 tx )
{
   float R1 = param0;
   float R2 = param1;
   float r  = param2*scale;
   float p  = 5;  // must be odd
   float q  = 3;
   
   float kl = (R1 + R2*cos(p*u) + r*cos(v));
   float x = kl*cos(q*u);
   float y = kl*sin(q*u);
   float z = r*sin(v) + R2*sin(p*u);
   
   vx   = vec4(x,y,z,1);
   tx   = vec2(u,v)*invPI*0.5;
   
   vec3 du = vec3(-kl*q*sin(q*u)-R2*p*sin(p*u)*cos(q*u),kl*q*cos(q*u)-R2*p*sin(p*u)*sin(q*u),p*R2*cos(p*u));
   vec3 dv = vec3(-r*sin(v)*cos(q*u),-r*sin(v)*sin(q*u),r*cos(v));
   
   ndir = normalize(cross(du,dv));
}
*/

// ----------------------------------------------------------
//
// ----------------------------------------------------------
void moebius( in float u, in float v, in float scale,
              out vec4 vx, out vec3 ndir, out vec2 tx )
{
   float n = param0;
   
   float r   = u;
   float phi = v;   
   r*=scale;
   
   float cosphi = cos(phi);
   float sinphi = sin(phi);
   float cp = cos(0.5*n*phi);
   float sp = sin(0.5*n*phi);
   
   float kl = 1.0+0.5*r*cp;
   float x = cos(phi)*kl;
   float y = sin(phi)*kl;
   float z = 0.5*r*sp;
   
   vx   = vec4(x,y,z,1);
   tx   = (vec2(u,v) - uv_min)/(uv_max - uv_min);
   
   vec3 dr   = vec3(0.5*cosphi*cp,0.5*sinphi*cp,0.5*sp);
   vec3 dphi = vec3(-sinphi*kl - 0.25*r*n*cosphi*sp,
                     cosphi*kl - 0.25*r*n*sinphi*sp,
                     0.25*r*n*cp);
                  
   ndir = normalize(cross(dr,dphi));
}


// ----------------------------------------------------------
//
// ----------------------------------------------------------
void graph( in float u, in float v, in float scale,
            out vec4 vx, out vec3 ndir, out vec2 tx )
{
   float n = param0;
   float m = param1;
   float a = param2;
   
   float su = sin(2*PI*n*u);
   float cu = cos(2*PI*n*u);
   float sv = sin(2*PI*m*v);
   float cv = cos(2*PI*m*v);
   
   float x = u;
   float y = v;
   float z = a*a*su*sv;
   
   vx   = vec4(x,y,z,1);
   tx   = (vec2(u,v) - uv_min)/(uv_max - uv_min);
   
   float nx = -2*PI*n*a*a*cu*sv;
   float ny = -2*PI*m*a*a*su*cv;
   float nz = 1.0;
   
   ndir = normalize(vec3(nx,ny,nz));
}

// ----------------------------------------------------------
//
// ----------------------------------------------------------
void main(void)
{
   vec4 vert = vec4(in_position.xyz,1.0);
   vec3 ndir = in_position.xyz;
      
   int wc = chart;
   
   float u = vert.x;
   float v = vert.y;
   float scale = 1.0;
   
   // construct surface vertices
   if (isSurface==1) {
      u = uv_min.x + (vert.x + (gl_InstanceID % slice.x))*uvStep.x;
      v = uv_min.y + (vert.y + (gl_InstanceID / slice.x))*uvStep.y;
   }
   // set particle vertices
   else if (isSurface==0) {
      u  = in_position.x;
      v  = in_position.y;
      wc = int(in_position.z);
      gl_PointSize = 4.0;
      scale = 1.01;
      color = in_color;
   }

   
   switch (objType)
   {
      case SURF_TYPE_PLANE: {
         plane(u,v,scale,vert,ndir,texCoords);
         break;
      }
      case SURF_TYPE_SPHERE: {
         sphere(u,v,scale,vert,ndir,texCoords);
         break;
      }
      case SURF_TYPE_ELLIPSOID: {
         ellipsoid(u,v,scale,vert,ndir,texCoords);
         break;
      }
      case SURF_TYPE_FRUSTUM: {
         frustum(u,v,scale,vert,ndir,texCoords);
         break;
      }
      case SURF_TYPE_TORUS: {
         torus(u,v,scale,vert,ndir,texCoords);
         break;
      }
      case SURF_TYPE_MOEBIUS: {
         moebius(u,v,scale,vert,ndir,texCoords);
         break;
      }
      case SURF_TYPE_GRAPH: {
         graph(u,v,scale,vert,ndir,texCoords);
         break;
      }
   }
   
   doDiscardVertex = 0;
   //if (vert.w<0.5)  doDiscardVertex = 1;
      
   vert.xyz = vert.x*e1 + vert.y*e2 + vert.z*e3 + center;
   localPos = vert.xyz;
   normDir  = ndir.x*e1 + ndir.y*e2 + ndir.z*e3;
   gl_Position = proj_matrix * view_matrix * vert;
}
