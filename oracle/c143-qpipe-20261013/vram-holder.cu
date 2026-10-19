// Holds all but KEEP_MB of free device memory until killed, to test the
// --q-pipeline admission fallback.
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>
int main(int argc, char **argv) {
    size_t keep = (size_t)(argc > 1 ? atol(argv[1]) : 2000) << 20, f, t;
    cudaMemGetInfo(&f, &t);
    size_t take = f > keep ? f - keep : 0;
    void *p = nullptr;
    if (cudaMalloc(&p, take) != cudaSuccess) { printf("alloc failed\n"); return 1; }
    printf("holding %.2f GB, %.2f GB left\n", take / 1073741824.0, keep / 1073741824.0);
    fflush(stdout);
    pause();
    return 0;
}
