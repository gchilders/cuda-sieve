#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <sys/time.h>
static double now(){struct timeval t;gettimeofday(&t,0);return t.tv_sec+t.tv_usec*1e-6;}
int main(int argc,char**argv){@autoreleasepool{
  id<MTLDevice> d=MTLCreateSystemDefaultDevice(); NSError*e=nil;
  id<MTLLibrary> l=[d newLibraryWithURL:[NSURL fileURLWithPath:@"spin.metallib"] error:&e];
  id<MTLComputePipelineState> p=[d newComputePipelineStateWithFunction:[l newFunctionWithName:@"k_spin"] error:&e];
  id<MTLCommandQueue> q=[d newCommandQueue];
  id<MTLBuffer> o=[d newBufferWithLength:64 options:MTLResourceStorageModeShared];
  for (uint32_t it = 1u<<27; it <= (1u<<30); it <<= 1) {
    id<MTLCommandBuffer> cb=[q commandBuffer];
    id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
    [en setComputePipelineState:p];[en setBuffer:o offset:0 atIndex:0];
    [en setBytes:&it length:4 atIndex:1];
    [en dispatchThreadgroups:MTLSizeMake(64,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
    [en endEncoding];
    double t0=now(); [cb commit]; [cb waitUntilCompleted]; double dt=now()-t0;
    printf("iters=%-10u  %6.2f s  %s\n", it, dt,
           cb.error ? cb.error.localizedDescription.UTF8String : "completed");
    if (cb.error) break;
    if (dt > 60.0) { printf("(stopping: already past 60 s with no abort)\n"); break; }
  }
}return 0;}
