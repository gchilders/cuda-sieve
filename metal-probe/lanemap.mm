#import <Metal/Metal.h>
#include <cstdio>
int main(){@autoreleasepool{
  id<MTLDevice> d=MTLCreateSystemDefaultDevice(); NSError*e=nil;
  id<MTLLibrary> l=[d newLibraryWithURL:[NSURL fileURLWithPath:@"lanemap.metallib"] error:&e];
  id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:@"k_lanemap"] error:&e];
  id<MTLCommandQueue> q=[d newCommandQueue];
  const int N=512;
  id<MTLBuffer> o=[d newBufferWithLength:N*3*4 options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb=[q commandBuffer]; id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
  [en setComputePipelineState:p]; [en setBuffer:o offset:0 atIndex:0];
  [en dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(N,1,1)];
  [en endEncoding];[cb commit];[cb waitUntilCompleted];
  uint32_t*r=(uint32_t*)o.contents; int badlane=0, badsg=0;
  for(int i=0;i<N;i++){
    if(r[i*3+1] != (uint32_t)(i & 31)) badlane++;
    if(r[i*3+2] != (uint32_t)(i >> 5)) badsg++;
  }
  printf("threads=%d\n", N);
  printf("lane  != (tid & 31) : %d mismatches\n", badlane);
  printf("simdgroup != (tid>>5): %d mismatches\n", badsg);
  printf("%s\n", (badlane||badsg) ? "LANE MAPPING DIFFERS FROM CUDA'S ASSUMPTION"
                                  : "lane mapping matches CUDA's assumption");
}return 0;}
