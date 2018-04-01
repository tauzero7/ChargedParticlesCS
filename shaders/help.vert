layout(location = 0) in vec4 in_position;
uniform mat4 proj_matrix;
uniform vec2 origin;
uniform vec2 size;
void main(void)
{
   vec2 vert = origin + in_position.xy*size;
   gl_Position = proj_matrix * vec4(vert,-0.3,1.0);
}
