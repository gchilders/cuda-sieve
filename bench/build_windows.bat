@echo off
setlocal EnableExtensions

rem Native Windows build for the main CUDA sieve executable.
rem Run from an "x64 Native Tools Command Prompt for VS" with CUDA on PATH.
rem GPU_ARCH may be "all" (default) or a bare compute capability, e.g. 86/120.

where cl >nul 2>nul || (
    echo error: cl.exe not found. Run this from an x64 Visual Studio Native Tools prompt.
    exit /b 1
)
where nvcc >nul 2>nul || (
    echo error: nvcc.exe not found. Add the CUDA Toolkit bin directory to PATH.
    exit /b 1
)

if not defined GPU_ARCH set "GPU_ARCH=all"
if /I "%GPU_ARCH%"=="all" (
    set "NVCC_ARCH=-gencode arch=compute_120,code=sm_120 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80"
) else (
    set "NVCC_ARCH=-gencode arch=compute_%GPU_ARCH%,code=sm_%GPU_ARCH%"
)

echo Building host C objects with cl.exe...
for %%F in (fb_load.c verify_cpu.c poly.c primes.c rfb.c fb_cado.c platform.c runlog.c) do (
    cl /nologo /O2 /W3 /std:c11 /D_CRT_SECURE_NO_WARNINGS /c %%F || exit /b 1
)

rem The main executable needs only fbgen's streaming special-q generator.
rem FBGEN_LIBRARY excludes the standalone pthread/open_memstream writer.
cl /nologo /O2 /W3 /std:c11 /D_CRT_SECURE_NO_WARNINGS /DFBGEN_LIBRARY /c fbgen.c /Fofbgen_lib.obj || exit /b 1
cl /nologo /O2 /W3 /EHsc /std:c++17 /D_CRT_SECURE_NO_WARNINGS /c boinc_support.cpp || exit /b 1

echo Building CUDA objects with cl.exe as nvcc's host compiler...
nvcc %NVCC_ARCH% --threads 0 -O3 -std=c++17 -lineinfo -ccbin cl ^
    -Xcompiler "/O2 /W3 /EHsc /D_CRT_SECURE_NO_WARNINGS" ^
    -c bench_main.cu -o bench_main.obj || exit /b 1
nvcc %NVCC_ARCH% --threads 0 -O3 -std=c++17 -lineinfo -ccbin cl ^
    -Xcompiler "/O2 /W3 /EHsc /D_CRT_SECURE_NO_WARNINGS" ^
    -c bench_kernels.cu -o bench_kernels.obj || exit /b 1

echo Linking bench.exe...
nvcc %NVCC_ARCH% --cudart static -ccbin cl -Xlinker "/OPT:REF" -o bench.exe ^
    bench_main.obj bench_kernels.obj fb_load.obj verify_cpu.obj poly.obj ^
    primes.obj rfb.obj fb_cado.obj platform.obj runlog.obj fbgen_lib.obj ^
    boinc_support.obj || exit /b 1

echo Built %CD%\bench.exe
exit /b 0
