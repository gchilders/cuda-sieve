#import <Metal/Metal.h>
#include <cstdio>
#include <cstring>
#include <cmath>
static uint32_t f2b(float f){uint32_t b;memcpy(&b,&f,4);return b;}
static float b2f(uint32_t b){float f;memcpy(&f,&b,4);return f;}
static uint64_t rs=0x9E3779B97F4A7C15ull;
static uint64_t rnd(){rs^=rs>>12;rs^=rs<<25;rs^=rs>>27;return rs*0x2545F4914F6CDD1Dull;}
int main(){@autoreleasepool{
  id<MTLDevice> d=MTLCreateSystemDefaultDevice(); NSError*e=nil;
  id<MTLLibrary> l=[d newLibraryWithURL:[NSURL fileURLWithPath:@"fp32.metallib"] error:&e];
  if(!l){printf("lib: %s\n",e.description.UTF8String);return 1;}
  id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:@"k_fp32"] error:&e];
  id<MTLCommandQueue> q=[d newCommandQueue];
  const int N=1<<20;
  id<MTLBuffer> ba=[d newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bb=[d newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> bc=[d newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> od=[d newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
  id<MTLBuffer> of=[d newBufferWithLength:N*4 options:MTLResourceStorageModeShared];
  float*A=(float*)ba.contents,*B=(float*)bb.contents,*C=(float*)bc.contents;
  for(int i=0;i<N;i++){
    uint32_t ea=1+(rnd()%250), eb=1+(rnd()%250), ec=1+(rnd()%250);
    A[i]=b2f((uint32_t)((rnd()&1)<<31)|(ea<<23)|(uint32_t)(rnd()&0x7fffff));
    B[i]=b2f((uint32_t)((rnd()&1)<<31)|(eb<<23)|(uint32_t)(rnd()&0x7fffff));
    C[i]=b2f((uint32_t)((rnd()&1)<<31)|(ec<<23)|(uint32_t)(rnd()&0x7fffff));
  }
  id<MTLCommandBuffer> cb=[q commandBuffer];id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
  [en setComputePipelineState:p];[en setBuffer:od offset:0 atIndex:0];[en setBuffer:of offset:0 atIndex:1];
  [en setBuffer:ba offset:0 atIndex:2];[en setBuffer:bb offset:0 atIndex:3];[en setBuffer:bc offset:0 atIndex:4];
  [en dispatchThreadgroups:MTLSizeMake(N/256,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
  [en endEncoding];[cb commit];[cb waitUntilCompleted];
  float*GD=(float*)od.contents,*GF=(float*)of.contents;
  long nd=0,nf=0,nd_sub=0,nf_sub=0,nd_norm=0,nf_norm=0; int maxd_norm=0;
  /* "subnormal-involved" = host result OR gpu result is subnormal/zero */
  #define SUBZ(b) ((((b)>>23)&0xff)==0)
  for(int i=0;i<N;i++){
    float wd=A[i]/B[i], wf=fmaf(A[i],B[i],C[i]);
    uint32_t gb,wb;
    gb=f2b(GD[i]); wb=f2b(wd);
    if(gb!=wb){nd++; if(SUBZ(gb)||SUBZ(wb))nd_sub++; else {nd_norm++; int u=abs((int)gb-(int)wb); if(u>maxd_norm)maxd_norm=u;}}
    gb=f2b(GF[i]); wb=f2b(wf);
    if(gb!=wb){nf++; if(SUBZ(gb)||SUBZ(wb))nf_sub++; else nf_norm++;}
  }
  printf("fp32 divide  : %ld/%d differ -- %ld involve a subnormal/zero, %ld between NORMALS (max %d ULP)\n",nd,N,nd_sub,nd_norm,maxd_norm);
  printf("fp32 fma     : %ld/%d differ -- %ld involve a subnormal/zero, %ld between NORMALS\n",nf,N,nf_sub,nf_norm);
}return 0;}
