
uniform int   isSurface;
uniform int   objID;
uniform ivec2 slice;

uniform int   useFixedColor;
uniform vec3  fixedColor;
uniform int   whichShading;
uniform int   showsplats;
uniform vec2  fboOffset;
uniform vec2  uv_min;
uniform vec2  uv_max;
uniform ivec2 useMod;

uniform vec3  obsPos;

uniform sampler2D fboTex;
uniform sampler2D fboPotiTex;
uniform int showPoti;

in vec2 texCoords;
in vec3 localPos;
in vec3 normDir;
in vec4 color;
flat in int  doDiscardVertex;

layout(location = 0) out vec4 out_frag_color;

float Frequency = 20;

// ----------------------------------------------------------
//
// ----------------------------------------------------------
vec3 checker()
{
   float valX = sign(sin(texCoords.x*2*PI*Frequency));
   float valY = sign(sin(texCoords.y*PI*Frequency));
   return vec3(clamp(valX*valY,0.0,1.0));
}

// ----------------------------------------------------------
//
// ----------------------------------------------------------
//http://www.yaldex.com/open-gl/ch17lev1sec5.html
vec3 softChecker()
{
   vec3 Color1 = vec3(0,0,0);
   vec3 Color2 = vec3(1,1,1);
   vec3 AvgColor = vec3(0.5);
   vec3 color;
   // Determine the width of the projection of one pixel into s-t space
   vec2 fw = fwidth(texCoords);

   // Determine the amount of fuzziness
   
   vec2 fuzz = Frequency * 2 * fw;

   float fuzzMax = max(fuzz.s, fuzz.t);

   // Determine the position in the checkerboard pattern
   vec2 checkPos = fract(texCoords * Frequency * vec2(1,0.5));
   if (fuzzMax < 0.5)
   {
      // If the filter width is small enough, compute the pattern color
      vec2 p = smoothstep(vec2(0.5), fuzz + vec2(0.5), checkPos) +
               (1.0 - smoothstep(vec2(0.0), fuzz, checkPos));
      color = mix(Color1, Color2, p.x * p.y + (1.0 - p.x) * (1.0 - p.y));

      // Fade in the average color when we get close to the limit
      color = mix(color, AvgColor, smoothstep(0.125, 0.5, fuzzMax));
    }
    else
    {
      // Otherwise, use only the average color
      color = AvgColor;
    }
   return color;
}

// ----------------------------------------------------------
//
// ----------------------------------------------------------
vec3 softLattice()
{
   float sx = sin(texCoords.x*2*PI*Frequency);
   float sy = sin(texCoords.y*PI*Frequency);
   float lf = 128.0;
   sx = pow(abs(sx),lf);
   sy = pow(abs(sy),lf);
   float valX = clamp(1.0-sx,0.0,1.0);
   float valY = clamp(1.0-sy,0.0,1.0);
   return vec3(valX*valY);
}

// ----------------------------------------------------------
//
// ----------------------------------------------------------
void main(void)
{
   float lambert = clamp(abs(dot(normalize(obsPos-localPos),normDir)),0.0,1.0);
   float distOP  = length(obsPos-localPos);
       
   out_frag_color = vec4(1,0,0,1);     
   if (isSurface==1) {   
      if (objID==2) {
         float u = texCoords.x;
         float v = texCoords.y;
         if (u*u+v*v>1+1e-4)
            discard;
      }
      
      float dampFac = 0.4;
      //lambert = 1;
      switch (whichShading) 
      {
         case 0:
            out_frag_color.rgb = checker()*lambert*dampFac;
            break;
         case 1:
            out_frag_color.rgb = softChecker()*lambert*dampFac;
            break;
         case 2:
            out_frag_color.rgb = softLattice()*lambert*dampFac;
            break;
         default:
         case 3:
            out_frag_color.rgb = vec3(lambert)*dampFac;
            //out_frag_color.rgb = vec3(abs(lambert)*0.4);
            break;
      }
   }
   else {
      out_frag_color = color;
   }
   
   if (useFixedColor==1) {
      out_frag_color.rgb = fixedColor;
   }

   if ((showsplats==1 || showsplats==2) && showPoti!=1)
   {
      // lambert = 1.0;
      vec2 tc = texCoords;    // [0,1] x [0,1]
      vec2 xy = (uv_max - uv_min)*tc + uv_min;
      vec2 sc = (xy - uv_min + fboOffset)/(uv_max - uv_min + 2*fboOffset);
      out_frag_color += texture2D(fboTex,sc)*lambert;

#if 1   // use FBO's extended domain
      float x2 = (uv_max.x - uv_min.x)*tc.x + uv_max.x;
      if (x2<uv_max.x+fboOffset.x && useMod[0]==1) {
         float s2 = (x2 - uv_min.x + fboOffset.x)/(uv_max.x - uv_min.x + 2*fboOffset.x);
         out_frag_color += texture2D(fboTex,vec2(s2,sc.y))*lambert;
      }
      
      float x3 = (uv_max.x - uv_min.x)*tc.x + 2.0*uv_min.x - uv_max.x;
      if (x3>uv_min.x-fboOffset.x && useMod[0]==1) {
         float s3 = (x3 - uv_min.x + fboOffset.x)/(uv_max.x - uv_min.x + 2*fboOffset.x);
         out_frag_color += texture2D(fboTex,vec2(s3,sc.y))*lambert;
      }
      
      float y2 = (uv_max.y - uv_min.y)*tc.y + uv_max.y;
      if (y2<uv_max.y+fboOffset.y && useMod[1]==1) {
         float s2 = (y2 - uv_min.y + fboOffset.y)/(uv_max.y - uv_min.y + 2*fboOffset.y);
         out_frag_color += texture2D(fboTex,vec2(sc.x,s2))*lambert;
      }
      
      float y3 = (uv_max.y - uv_min.y)*tc.y + 2.0*uv_min.y - uv_max.y;
      if (y3>uv_min.y-fboOffset.y && useMod[1]==1) 
      {
         float s3 = (y3 - uv_min.y + fboOffset.y)/(uv_max.y - uv_min.y + 2*fboOffset.y);
         out_frag_color += texture2D(fboTex,vec2(sc.x,s3))*lambert;
      }
#endif
   }
   
   if (showPoti==1)
   {
      vec2 tc = texCoords;
      vec3 rgb = texture2D(fboPotiTex,tc).rgb;
      //rgb = log(rgb);
      //out_frag_color.rgb = vec3(1)-abs(rgb)*6e-4;
      out_frag_color.rgb = abs(rgb)*3e-6;
   }
      
   //out_frag_color = vec4(texCoords,0,1);
   //out_frag_color = vec4(0,0,0,1);
}
