
layout(location = 0) in vec4 in_position;

uniform mat4 proj_matrix;

out vec2 texCoords;


// ----------------------------------------------------------
//
// ----------------------------------------------------------
void main(void)
{
   vec4 vert = vec4(in_position.xy,0.0,1.0);   
   texCoords = in_position.xy;
   gl_Position = proj_matrix * vert;
}
