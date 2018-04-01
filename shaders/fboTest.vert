

layout(location = 0) in vec4 in_position;

uniform mat4 proj_matrix;
uniform vec2  uv_min;
uniform vec2  uv_max;
uniform vec2  offset;

out vec2 texCoords;

// ----------------------------------------------------------
//
// ----------------------------------------------------------
void main(void)
{
   vec4 vert = vec4(in_position.xy,0.0,1.0);
    texCoords   = vert.xy;
   vert.xy = (uv_max - uv_min + 2*offset)*vert.xy + uv_min - offset;   
  
   gl_Position = proj_matrix * vert;
}
