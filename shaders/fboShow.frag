
uniform sampler2D tex;

in vec2 texCoords;

layout(location = 0) out vec4 out_frag_color;


void main(void)
{
   out_frag_color = vec4(texCoords,0,1);
   out_frag_color = texture2D(tex,texCoords);
}
