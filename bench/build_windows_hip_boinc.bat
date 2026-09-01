@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem Native Windows HIP+BOINC build of the sieve benchmark. Same as
rem build_windows_hip.bat, but adds -DHAVE_BOINC/-DBENCH_HIP_BUILD and links
rem BOINC's client-side static libraries.
rem
rem Build those libraries first (a separate BOINC checkout, not this repo):
rem   cd C:\dev\boinc
rem   git clone https://github.com/BOINC/boinc.git .          (if not done yet)
rem   msbuild win_build\vcpkg_3rdparty_dependencies.vcxproj -p:Configuration=Release -p:Platform=x64
rem   msbuild win_build\libboinc.vcxproj -p:Configuration=Release -p:Platform=x64
rem   msbuild win_build\libboincapi.vcxproj -p:Configuration=Release -p:Platform=x64
rem
rem Knobs:
rem   BOINC_ROOT  C:\dev\boinc (default) | path to a BOINC checkout with the
rem               two libs above already built at
rem               <BOINC_ROOT>\win_build\Build\x64\Release\
rem   GFX_ARCH/CF_LMAX/DEFS: same meaning as build_windows_hip.bat
rem
rem   build_windows_hip_boinc.bat clean
rem
rem Produces bench_hip_boinc.exe, separate from build_windows_hip.bat's
rem bench_hip.exe, so neither build clobbers the other's output or object
rem files (all object names here carry a "_boinc" suffix for the same
rem reason). See CLAUDE.md for how this was validated.
rem ---------------------------------------------------------------------------

if /I "%~1"=="clean" goto :do_clean
if not "%~1"=="" (
    echo error: unknown argument "%~1". Use no argument or clean.
    exit /b 1
)

if not defined BOINC_ROOT set "BOINC_ROOT=C:\dev\boinc"
set "BOINC_LIBDIR=%BOINC_ROOT%\win_build\Build\x64\Release"

if not exist "%BOINC_LIBDIR%\libboinc.lib" (
    echo error: %BOINC_LIBDIR%\libboinc.lib not found. Build BOINC's client
    echo        libraries first -- see the comment at the top of this script.
    exit /b 1
)
if not exist "%BOINC_LIBDIR%\libboincapi.lib" (
    echo error: %BOINC_LIBDIR%\libboincapi.lib not found. Build BOINC's client
    echo        libraries first -- see the comment at the top of this script.
    exit /b 1
)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo error: vcvars64.bat failed. Is VS 2022 Build Tools with the
    echo        VCTools workload installed at the path in this script?
    exit /b 1
)

where cl >nul 2>nul || (
    echo error: cl.exe not found even after vcvars64.bat.
    exit /b 1
)

set "HIP_PATH=C:\rocm"
set "ROCM_PATH=C:\rocm"
set "PATH=%PATH%;C:\rocm\bin"
where hipcc >nul 2>nul || (
    echo error: hipcc.exe not found on C:\rocm\bin. Check the ROCm install.
    exit /b 1
)

rem GFX_ARCH=all builds the same RDNA1-4 fat binary as build_windows_hip.bat
rem (consumer/wavefront32 parts only -- see that script's header comment for
rem the full target list and rationale). This is the knob a real BOINC
rem deployment binary should use, since a volunteer pool spans many cards.
if not defined GFX_ARCH set "GFX_ARCH=gfx1103"
if /I "%GFX_ARCH%"=="all" (
    set "HIP_ARCH=--offload-arch=gfx1010 --offload-arch=gfx1012 --offload-arch=gfx1030 --offload-arch=gfx1031 --offload-arch=gfx1032 --offload-arch=gfx1034 --offload-arch=gfx1100 --offload-arch=gfx1101 --offload-arch=gfx1102 --offload-arch=gfx1103 --offload-arch=gfx1150 --offload-arch=gfx1151 --offload-arch=gfx1200 --offload-arch=gfx1201"
) else (
    set "HIP_ARCH=--offload-arch=%GFX_ARCH%"
)
set "HIP_DEVLIB=--rocm-device-lib-path=C:\rocm\lib\llvm\amdgcn\bitcode"

rem ---- CF_LMAX -----------------------------------------------------------
rem Duplicated from build_windows_hip.bat/build_windows.bat/bench/Makefile
rem (build_windows.bat must not be touched by this port -- see CLAUDE.md).
rem Update all four in lockstep if this ever changes.
if not defined CF_LMAX set "CF_LMAX=4"
if "%CF_LMAX%"=="3" goto :cflmax_ok
if "%CF_LMAX%"=="4" goto :cflmax_ok
echo error: CF_LMAX must be 3 (96-bit cofactors) or 4 (128-bit) -- got "%CF_LMAX%".
exit /b 1
:cflmax_ok
set "CF_LMAX_DEF=-DCF_LMAX=%CF_LMAX%"

rem ---- build stamp ---------------------------------------------------------
rem Same duplication note as CF_LMAX above.
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

rem ---- BOINC and build flags ----------------------------------------------
rem BENCH_HIP_BUILD: read by boinc_support.cpp (only under HAVE_BOINC) to
rem pick BOINC's "ATI" vendor string instead of "NVIDIA" when checking a
rem client GPU assignment. See boinc_support.cpp and CLAUDE.md.
set "BOINC_INC=/I "%BOINC_ROOT%\api" /I "%BOINC_ROOT%\lib""
set "BOINC_DEF=-DHAVE_BOINC -DBENCH_HIP_BUILD"
set "BOINC_LIBS="%BOINC_LIBDIR%\libboincapi.lib" "%BOINC_LIBDIR%\libboinc.lib""

rem BOINC_INC only goes on CXXFLAGS: boinc_support.cpp is the only
rem translation unit that #includes any BOINC header (boinc_api.h) directly
rem -- bench_main_hip.cpp/bench_kernels.hip/fbgen_gpu.hip only call the
rem bench_boinc_* wrapper functions declared in bench.h. It's also MSVC-only
rem syntax (/I "path"), which hipcc's clang driver (used for HIPFLAGS) does
rem not accept -- clang wants -I, not /I -- so leaving it out of HIPFLAGS is
rem both unnecessary and would be a syntax error there.
set "CFLAGS=/nologo /O2 /W3 /MT -D_CRT_SECURE_NO_WARNINGS %BOINC_DEF% %CF_LMAX_DEF% %DEFS%"
set "CXXFLAGS=/nologo /O2 /W3 /MT /EHsc -D_CRT_SECURE_NO_WARNINGS %BOINC_DEF% %BOINC_INC% %CF_LMAX_DEF% %DEFS%"
set "HIPFLAGS=-O2 -std=c++17 %HIP_ARCH% %HIP_DEVLIB% -D_CRT_SECURE_NO_WARNINGS %BOINC_DEF% %CF_LMAX_DEF% %DEFS%"

echo Building host C objects with cl.exe... (GFX_ARCH=%GFX_ARCH% CF_LMAX=%CF_LMAX% HAVE_BOINC=1 build=%GIT_DESC%)
for %%F in (fb_load.c verify_cpu.c poly.c primes.c rfb.c fb_cado.c platform.c) do (
    cl %CFLAGS% /std:c11 /c %%F || exit /b 1
)

cl %CFLAGS% /std:c11 -DBENCH_GIT_DESC=\"%GIT_DESC%\" -DBENCH_DEFS=\"%DEFS%\" /c runlog.c || exit /b 1
cl %CFLAGS% /std:c11 -DFBGEN_LIBRARY /c fbgen.c /Fofbgen_lib_boinc.obj || exit /b 1
cl %CXXFLAGS% /std:c++17 /c boinc_support.cpp /Foboinc_support_boinc.obj || exit /b 1

echo Building HIP objects with hipcc (cl.exe as host compiler)...
hipcc %HIPFLAGS% -c bench_main_hip.cpp -o bench_main_hip_boinc.obj || exit /b 1
hipcc %HIPFLAGS% -c bench_kernels.hip -o bench_kernels_hip_boinc.obj || exit /b 1
hipcc %HIPFLAGS% -DFBGEN_GPU_LIBRARY -c fbgen_gpu.hip -o fbgen_gpu_hip_lib_boinc.obj || exit /b 1

echo Linking bench_hip_boinc.exe...
hipcc %HIP_ARCH% %HIP_DEVLIB% -o bench_hip_boinc.exe ^
    bench_main_hip_boinc.obj bench_kernels_hip_boinc.obj fbgen_gpu_hip_lib_boinc.obj fb_load.obj verify_cpu.obj poly.obj ^
    primes.obj rfb.obj fb_cado.obj platform.obj runlog.obj fbgen_lib_boinc.obj ^
    boinc_support_boinc.obj %BOINC_LIBS% ^
    -lWs2_32 -lwininet -lpsapi -lPowrprof -lIphlpapi -lAdvapi32 -lUser32 -lCrypt32 -lShell32 -lVersion || exit /b 1

echo Built %CD%\bench_hip_boinc.exe (GFX_ARCH=%GFX_ARCH% CF_LMAX=%CF_LMAX% build=%GIT_DESC%)
echo NOTE: bench_hip_boinc.exe needs C:\rocm\bin on PATH (libamdhip64.dll etc.)
echo       to run -- same as bench_hip.exe, no static HIP runtime exists.
exit /b 0

:do_clean
rem Deliberately does NOT touch the shared host C objects (fb_load.obj,
rem platform.obj, etc.), bench.exe, or bench_hip.exe/its objects -- those
rem belong to build_windows.bat and build_windows_hip.bat.
del /q *_boinc.obj bench_hip_boinc.exe 2>nul
echo Cleaned Windows HIP+BOINC build products in %CD%
exit /b 0
