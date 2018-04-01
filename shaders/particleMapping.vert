
layout(location = 0) in vec4 in_position;
layout(location = 2) in vec4 in_color;

uniform mat4  proj_matrix;
uniform vec2  uv_mod;
uniform ivec2 useMod;

out vec2 v_texCoords;
out vec4 v_color;


float mmod( float x, float y )
{
   if (y==0.0)
      return x;     
   return x - floor(x/y)*y;
}

// ----------------------------------------------------------
//
// ----------------------------------------------------------
void main(void)
{
   vec4 vert = vec4(in_position.xy,0.0,1.0);
   
   if (useMod[0]==1)
      vert.x = mmod(vert.x,uv_mod.x);
   if (useMod[1]==1)
      vert.y = mmod(vert.y,uv_mod.y);
   
   v_texCoords = in_position.xy;   
   v_color     = in_color;
   
   gl_Position = vert;
}
