
uniform vec2  uv_min;
uniform vec2  uv_max;
uniform vec2  offset;

in vec2 texCoords;

layout(location = 0) out vec4 out_frag_color;

void main(void)
{
   vec4 color = vec4(0.3,0,0,1);
   
   vec2 tc = (uv_max - uv_min + 2*offset)*texCoords + uv_min - offset;
   
   if (tc.x>=uv_min.x && tc.x<=uv_max.x && tc.y>=uv_min.y && tc.y<=uv_max.y)  {
      color = vec4(vec3(0.2),1);
   }
   else {
      color = vec4(0.2,0,0,1);
   }
   
   out_frag_color = color;
}
