@echo off
setlocal EnableExtensions

rem Native Windows build for the main CUDA sieve executable and, optionally,
rem the standalone GPU factor-base cache generator.
rem Run from an "x64 Native Tools Command Prompt for VS" with CUDA on PATH.
rem
rem Knobs, matching bench/Makefile so a Windows binary is the same build:
rem   GPU_ARCH  all (default) | native | a bare compute capability, e.g. 86
rem   CF_LMAX   4 (default, 128-bit cofactors) | 3 (96-bit)
rem   DEFS      extra -D's for pricing experiments, e.g. -DNORM_FAST_LOG2
rem             (dash form: these reach nvcc too, which rejects /D)
rem
rem   set GPU_ARCH=86 ^&^& set CF_LMAX=3 ^&^& build_windows.bat
rem   build_windows.bat fbgen_gpu   rem also build fbgen_gpu.exe
rem
rem Default builds bench.exe only.  The fbgen_gpu argument additionally builds
rem the reusable roots-file utility without making every ordinary build compile
rem fbgen_gpu.cu twice.  "build_windows.bat clean" removes both executables.

if /I "%~1"=="clean" goto :do_clean
set "BUILD_FBGEN_GPU="
if "%~1"=="" goto :arg_done
if /I "%~1"=="fbgen_gpu" (
    set "BUILD_FBGEN_GPU=1"
    goto :arg_done
)
echo error: unknown target "%~1". Use no argument, fbgen_gpu, or clean.
exit /b 1
:arg_done

where cl >nul 2>nul || (
    echo error: cl.exe not found. Run this from an x64 Visual Studio Native Tools prompt.
    exit /b 1
)
where nvcc >nul 2>nul || (
    echo error: nvcc.exe not found. Add the CUDA Toolkit bin directory to PATH.
    exit /b 1
)

rem ---- GPU_ARCH ----------------------------------------------------------
rem Validated rather than pasted straight into the gencode string. An
rem unchecked value produces "compute_sm_86" or "compute_8.6" and an opaque
rem ptxas diagnostic; the Makefile rejects the same three spellings by hand
rem and says why, so this does too.
if not defined GPU_ARCH set "GPU_ARCH=all"
if /I "%GPU_ARCH%"=="all" goto :arch_all
if /I "%GPU_ARCH%"=="native" goto :arch_native
echo %GPU_ARCH%| findstr /r /c:"^[0-9][0-9]*$" >nul || goto :arch_bad
set "NVCC_ARCH=-gencode arch=compute_%GPU_ARCH%,code=sm_%GPU_ARCH%"
goto :arch_done

:arch_bad
echo error: GPU_ARCH must be all, native, or a bare compute capability like 86
echo        or 120 -- got "%GPU_ARCH%". Not sm_86, not 8.6.
exit /b 1

:arch_all
rem Same fat-binary set as the Makefile's default, including the compute_80
rem PTX floor so a newer card than this list still runs.
set "NVCC_ARCH=-gencode arch=compute_120,code=sm_120 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80"
goto :arch_done

:arch_native
set "GPU_CC="
for /f "usebackq delims=" %%G in (`nvidia-smi --query-gpu^=compute_cap --format^=csv^,noheader 2^>nul`) do if not defined GPU_CC set "GPU_CC=%%G"
if not defined GPU_CC goto :native_bad
rem "8.6" -> "86"
set "GPU_CC=%GPU_CC:.=%"
set "GPU_CC=%GPU_CC: =%"
echo %GPU_CC%| findstr /r /c:"^[0-9][0-9]*$" >nul || goto :native_bad
set "NVCC_ARCH=-gencode arch=compute_%GPU_CC%,code=sm_%GPU_CC%"
goto :arch_done

:native_bad
echo error: GPU_ARCH=native: no usable compute capability from nvidia-smi.
echo        Pass GPU_ARCH=^<cc^>, e.g. GPU_ARCH=86, or GPU_ARCH=all.
exit /b 1

:arch_done

rem ---- CF_LMAX -----------------------------------------------------------
rem A build-shape knob, not a pricing define: it changes which splitter widths
rem exist, and a CF_LMAX=3 binary is shippable. See bench/Makefile.
if not defined CF_LMAX set "CF_LMAX=4"
if "%CF_LMAX%"=="3" goto :cflmax_ok
if "%CF_LMAX%"=="4" goto :cflmax_ok
echo error: CF_LMAX must be 3 (96-bit cofactors) or 4 (128-bit) -- got "%CF_LMAX%".
exit /b 1
:cflmax_ok
rem Dash form deliberately: cl.exe accepts -D as happily as /D, and nvcc
rem accepts only -D. One spelling that works for both keeps the CUDA and host
rem compiles from drifting apart on the one define that changes build shape.
set "CF_LMAX_DEF=-DCF_LMAX=%CF_LMAX%"

rem ---- build stamp -------------------------------------------------------
rem What makes a run log traceable to a source tree, and what keeps a pricing
rem build's relations distinguishable from production output. Without these
rem two defines runlog.c falls back to "unknown", and every Windows run log
rem claims an unknown commit whether or not that is true.
set "GIT_RAW="
for /f "usebackq delims=" %%G in (`git -C . describe --always --abbrev^=8 2^>nul`) do if not defined GIT_RAW set "GIT_RAW=%%G"
if not defined GIT_RAW goto :stamp_unknown
set "GIT_DIRT="
for /f "usebackq delims=" %%G in (`git -C . status --porcelain 2^>nul`) do set "GIT_DIRT=1"
if defined GIT_DIRT set "GIT_RAW=%GIT_RAW%-dirty"
set "GIT_DESC=%GIT_RAW%"
goto :stamp_done
:stamp_unknown
set "GIT_DESC=unknown"
:stamp_done

rem ---- flags -------------------------------------------------------------
rem /MT, not the default /MD: bench/Makefile carries its C++ and GCC support
rem runtimes into distribution binaries for the same reason. A /MD bench.exe
rem hard-depends on VCRUNTIME140.dll, and a BOINC volunteer without the
rem matching Visual C++ redistributable fails the task with a loader error
rem before main() runs.
set "CFLAGS=/nologo /O2 /W3 /MT -D_CRT_SECURE_NO_WARNINGS %CF_LMAX_DEF% %DEFS%"
set "CXXFLAGS=/nologo /O2 /W3 /MT /EHsc -D_CRT_SECURE_NO_WARNINGS %CF_LMAX_DEF% %DEFS%"

echo Building host C objects with cl.exe... (GPU_ARCH=%GPU_ARCH% CF_LMAX=%CF_LMAX% build=%GIT_DESC%)
for %%F in (fb_load.c verify_cpu.c poly.c primes.c rfb.c fb_cado.c platform.c) do (
    cl %CFLAGS% /std:c11 /c %%F || exit /b 1
)

rem runlog.c alone carries the build stamp, exactly as in the Makefile.
cl %CFLAGS% /std:c11 -DBENCH_GIT_DESC=\"%GIT_DESC%\" -DBENCH_DEFS=\"%DEFS%\" /c runlog.c || exit /b 1

rem The main executable needs only fbgen's streaming special-q generator.
rem FBGEN_LIBRARY excludes the standalone pthread/open_memstream writer.
cl %CFLAGS% /std:c11 -DFBGEN_LIBRARY /c fbgen.c /Fofbgen_lib.obj || exit /b 1
cl %CXXFLAGS% /std:c++17 /c boinc_support.cpp || exit /b 1

echo Building CUDA objects with cl.exe as nvcc's host compiler...
nvcc %NVCC_ARCH% --threads 0 -O3 -std=c++17 -lineinfo -ccbin cl %CF_LMAX_DEF% %DEFS% ^
    -Xcompiler "/O2 /W3 /EHsc /MT -D_CRT_SECURE_NO_WARNINGS" ^
    -c bench_main.cu -o bench_main.obj || exit /b 1
nvcc %NVCC_ARCH% --threads 0 -O3 -std=c++17 -lineinfo -ccbin cl %CF_LMAX_DEF% %DEFS% ^
    -Xcompiler "/O2 /W3 /EHsc /MT -D_CRT_SECURE_NO_WARNINGS" ^
    -c bench_kernels.cu -o bench_kernels.obj || exit /b 1

rem Production bench needs the same in-process algebraic factor-base generator
rem as the Makefile build.  FBGEN_GPU_LIBRARY removes the standalone CLI, timers
rem and text writer, so omitting --fb1 has no extra serialization path.
nvcc %NVCC_ARCH% --threads 0 -O3 -std=c++17 -lineinfo -ccbin cl %CF_LMAX_DEF% %DEFS% ^
    -DFBGEN_GPU_LIBRARY ^
    -Xcompiler "/O2 /W3 /EHsc /MT -D_CRT_SECURE_NO_WARNINGS" ^
    -c fbgen_gpu.cu -o fbgen_gpu_lib.obj || exit /b 1

echo Linking bench.exe...
nvcc %NVCC_ARCH% --cudart static -ccbin cl -Xlinker "/OPT:REF" -o bench.exe ^
    bench_main.obj bench_kernels.obj fbgen_gpu_lib.obj fb_load.obj verify_cpu.obj poly.obj ^
    primes.obj rfb.obj fb_cado.obj platform.obj runlog.obj fbgen_lib.obj ^
    boinc_support.obj || exit /b 1

echo Built %CD%\bench.exe (GPU_ARCH=%GPU_ARCH% CF_LMAX=%CF_LMAX% build=%GIT_DESC%)

if not defined BUILD_FBGEN_GPU exit /b 0

echo Building standalone fbgen_gpu.exe...
nvcc %NVCC_ARCH% --threads 0 -O3 -std=c++17 -lineinfo -ccbin cl %CF_LMAX_DEF% %DEFS% ^
    -Xcompiler "/O2 /W3 /EHsc /MT -D_CRT_SECURE_NO_WARNINGS" ^
    -c fbgen_gpu.cu -o fbgen_gpu.obj || exit /b 1
nvcc %NVCC_ARCH% --cudart static -ccbin cl -Xlinker "/OPT:REF" -o fbgen_gpu.exe ^
    fbgen_gpu.obj fbgen_lib.obj fb_load.obj fb_cado.obj poly.obj primes.obj platform.obj || exit /b 1
echo Built %CD%\fbgen_gpu.exe
exit /b 0

:do_clean
del /q *.obj bench.exe fbgen_gpu.exe 2>nul
echo Cleaned Windows build products in %CD%
exit /b 0
