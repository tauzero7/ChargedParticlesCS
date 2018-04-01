
in vec2 g_texCoords;
in vec4 g_color;

layout(location = 0) out vec4 out_frag_color;

void main(void)
{
   float r = length(g_texCoords);
   float val = exp(-r*r*4.0);
   
   out_frag_color = g_color*vec4(val);  
   
   //out_frag_color = vec4(0.2,0.2,0.5,1.0)*vec4(val);  
   //out_frag_color = vec4(0.3,0.3,0.8,1);
}
