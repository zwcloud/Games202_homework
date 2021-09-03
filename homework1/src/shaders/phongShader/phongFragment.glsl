#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 25
#define NUM_SAMPLES_SQRT 5
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
  const vec4 bitShift = vec4(1.0, 1.0/255.0, 1.0/65025.0, 1.0/16581375.0);
  float depth = dot(rgbaDepth, bitShift);
  return depth;
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

void averageSamples(){
  const int n = NUM_SAMPLES_SQRT;
  for( int y = -n; y < n; y++ ) {
    for( int x = -n; x < n; x++ ) {
      poissonDisk[y*n+x] = vec2(x,y)/float(n);
    }
  }
}

float PCF(sampler2D shadowMap, vec4 shadowCoord, float filterSize){
  float fragDepth = shadowCoord.z;
  poissonDiskSamples(shadowCoord.xy);
  float finalVisibility = 0.0;
  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    vec2 sampleCoord = poissonDisk[i] * filterSize + shadowCoord.xy;
    float sampleDepth = unpack(texture2D(shadowMap, sampleCoord));
    if(fragDepth < sampleDepth+0.005){
      finalVisibility += 1.0;
    }
  }
  finalVisibility = finalVisibility / float(NUM_SAMPLES);
  return finalVisibility;
}

float PCF_avg(sampler2D shadowMap, vec4 shadowCoord, float filterSizeUV){
  float fragDepth = shadowCoord.z;
  float finalVisibility = 0.0;
  for (int x = -8; x <= 8; x++) {
    for (int y = -8; y <= 8; y++) {
        vec2 offset = filterSizeUV * vec2(float(x),float(y))/16.0;
        vec2 sampleCoord = offset + shadowCoord.xy;
        float sampleDepth = unpack(texture2D(shadowMap, sampleCoord));
        if(fragDepth < sampleDepth+0.005) {
          finalVisibility += 1.0;
        }
    }
  }
  finalVisibility = finalVisibility / 49.0;
  return finalVisibility;
}

#define wLight 5.0

float findBlocker( sampler2D shadowMap, vec2 uv, float zReceiver, out float blockerNum) {
  float blockerDepthAvg = 0.0;
  blockerNum = 0.0;
  float texelSizeUV = 1.0 / 2048.0;
  float findSizeUV = 25.0*texelSizeUV;
  for (int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; i++) {
    vec2 sampleCoord = poissonDisk[i] * findSizeUV + uv;
    float sampleDepth = unpack(texture2D(shadowMap, sampleCoord));
    if(zReceiver > sampleDepth+0.001) {
      blockerDepthAvg += sampleDepth;
      blockerNum++;
    }
  }
  blockerDepthAvg /= blockerNum;
	return blockerDepthAvg;
}

float PCSS(sampler2D shadowMap, vec4 coords){

  float texelSizeUV = 1.0 / 2048.0;
  poissonDiskSamples(coords.xy);

  // STEP 1: avgblocker depth
  vec2 uv = coords.xy;
  float dReceiver = coords.z;
  float blockerNum = 0.0;
  float dBlocker = findBlocker(shadowMap, uv, dReceiver, blockerNum);
  if(blockerNum < 0.5) {//not covered by shadow
    return 1.0;
  }

  // STEP 2: penumbra size
  float wPenumbra = (dReceiver - dBlocker)*wLight/dBlocker;

  // STEP 3: filtering
  float filterSizeUV = 5.0*texelSizeUV*wPenumbra;
  float finalVisibility = PCF(shadowMap, coords, filterSizeUV);
  return finalVisibility;
}


float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){
  float fragDepth = shadowCoord.z;
  vec4 packedShadowMapDepth = texture2D(shadowMap, shadowCoord.xy);
  float shadowMapDepth = unpack(packedShadowMapDepth);
  if(fragDepth > shadowMapDepth+0.02){
    return 0.1;
  }
  else{
    return 1.0;
  }
}

vec3 blinnPhong(out vec3 ambient) {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {

  float visibility;

  //frag pos in light local space: vPositionFromLight
  //convert to NDC space
  vec3 shadowCoord = vPositionFromLight.xyz / vPositionFromLight.w;
  //convert to shadow map uv space (the screen space when rendering the shadow map)
  shadowCoord = shadowCoord * 0.5 + 0.5;
  //visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));
  //visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0), 5.0);
  
  float texelSizeUV = 1.0 / 2048.0;
  //visibility = PCF_avg(uShadowMap, vec4(shadowCoord, 1.0), texelSizeUV);

  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));

  vec3 ambient = vec3(0.0);
  vec3 phongColor = blinnPhong(ambient);

  gl_FragColor = vec4(ambient, 1.0) + vec4(phongColor * visibility, 1.0);
  //gl_FragColor = vec4(phongColor, 1.0);

  //float f = PCSS_check(uShadowMap, vec4(shadowCoord, 1.0));
  //gl_FragColor = vec4(f, f, f, 1.0);
}