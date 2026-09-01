@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem Native Windows HIP build of the sieve benchmark, for AMD GPUs. Mirrors
rem build_windows.bat's structure (same host C objects, same CF_LMAX/DEFS
rem knobs, same /MT static CRT rationale) but swaps nvcc for hipcc as the
rem device compiler, still using cl.exe as its host compiler via vcvars64.bat.
rem
rem Run from an ordinary (non-Administrator) PowerShell or cmd -- this script
rem calls vcvars64.bat itself, so it does NOT need to be run from an
rem "x64 Native Tools Command Prompt".
rem
rem Knobs:
rem   GFX_ARCH  gfx1103 (default, this box's Radeon 780M) | all (RDNA
rem             fat binary, see below) | any --offload-arch value
rem   CF_LMAX   4 (default, 128-bit cofactors) | 3 (96-bit)
rem   DEFS      extra -D's for pricing experiments (dash form: reaches hipcc too)
rem
rem   build_windows_hip.bat clean
rem
rem GFX_ARCH=all builds one fat binary covering RDNA1 through RDNA4 --
rem consumer/wavefront32 parts only (this codebase assumes warp width 32
rem throughout; CDNA's wavefront64 compute cards, e.g. MI100/MI200/MI300,
rem are excluded on purpose, matching CLAUDE.md's ground rule, not just
rem left out for convenience):
rem   gfx1010/1012        RDNA1: RX 5700/5700XT/5600XT, RX 5500/5500XT
rem   gfx1030/1031/1032/1034  RDNA2: RX 6800-6900 series, 6700(XT), 6600
rem                        series, 6500XT/6400
rem   gfx1100/1101/1102/1103  RDNA3: RX 7900 series, 7800XT/7700XT, 7600
rem                        series, and the 780M/880M iGPU family (this box)
rem   gfx1150/1151         RDNA3.5: Strix Point/Strix Halo iGPUs
rem   gfx1200/1201         RDNA4: RX 9060 series, RX 9070(XT)
rem UNLIKE the CUDA build's fat binary, there is no PTX-style forward-
rem compatible fallback here: HIP embeds real machine code per listed arch,
rem and only an EXACT match runs. A GPU whose gfx ID isn't in this list
rem (a future RDNA5 card, for instance) will not run this binary at all,
rem no matter how similar its ISA. Re-run with the new gfx ID added when
rem that happens. See CLAUDE.md for how this list was chosen and verified.
rem ---------------------------------------------------------------------------

if /I "%~1"=="clean" goto :do_clean
if not "%~1"=="" (
    echo error: unknown argument "%~1". Use no argument or clean.
    exit /b 1
)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo error: vcvars64.bat failed. Is VS 2022 Build Tools with the
    echo        VCTools workload installed at the path in this script?
    exit /b 1
)

where cl >nul 2>nul || (
    echo error: cl.exe not found even after vcvars64.bat. Check the VS
    echo        Build Tools installation.
    exit /b 1
)

set "HIP_PATH=C:\rocm"
set "ROCM_PATH=C:\rocm"
set "PATH=%PATH%;C:\rocm\bin"
where hipcc >nul 2>nul || (
    echo error: hipcc.exe not found on C:\rocm\bin. Check the ROCm install.
    exit /b 1
)

rem ---- GFX_ARCH ------------------------------------------------------------
if not defined GFX_ARCH set "GFX_ARCH=gfx1103"
if /I "%GFX_ARCH%"=="all" (
    set "HIP_ARCH=--offload-arch=gfx1010 --offload-arch=gfx1012 --offload-arch=gfx1030 --offload-arch=gfx1031 --offload-arch=gfx1032 --offload-arch=gfx1034 --offload-arch=gfx1100 --offload-arch=gfx1101 --offload-arch=gfx1102 --offload-arch=gfx1103 --offload-arch=gfx1150 --offload-arch=gfx1151 --offload-arch=gfx1200 --offload-arch=gfx1201"
) else (
    set "HIP_ARCH=--offload-arch=%GFX_ARCH%"
)

rem TheRock's Windows ROCm distribution puts the device bitcode library at
rem <ROCM_PATH>\lib\llvm\amdgcn\bitcode, not <ROCM_PATH>\amdgcn\bitcode where
rem clang looks by default -- without this flag, hipcc fails immediately with
rem "cannot find ROCm device library" before ever reaching any of this
rem project's own source. See CLAUDE.md.
set "HIP_DEVLIB=--rocm-device-lib-path=C:\rocm\lib\llvm\amdgcn\bitcode"

rem ---- CF_LMAX ---------------------------------------------------------------
rem This validation is intentionally identical to build_windows.bat's own
rem CF_LMAX block (itself already a batch reimplementation of bench/Makefile's
rem shell version) -- there is no shared-include mechanism between the two
rem .bat scripts here, and build_windows.bat must not be touched by this port
rem (see CLAUDE.md). If CF_LMAX's valid value set ever changes, update it in
rem THREE places: bench/Makefile, bench/build_windows.bat, and here.
if not defined CF_LMAX set "CF_LMAX=4"
if "%CF_LMAX%"=="3" goto :cflmax_ok
if "%CF_LMAX%"=="4" goto :cflmax_ok
echo error: CF_LMAX must be 3 (96-bit cofactors) or 4 (128-bit) -- got "%CF_LMAX%".
exit /b 1
:cflmax_ok
set "CF_LMAX_DEF=-DCF_LMAX=%CF_LMAX%"

rem ---- build stamp -----------------------------------------------------------
rem Same duplication note as CF_LMAX above: this block mirrors
rem build_windows.bat's git-describe/dirty-flag stamping (also already a copy
rem of the Makefile's GIT_RAW/GIT_DIRT/GIT_DESC logic). Update all three in
rem lockstep if this ever changes.
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
rem /MT for the same reason build_windows.bat uses it: a BOINC volunteer's
rem machine may not have the matching VC++ redistributable. NOTE this is
rem NOT the whole story for HIP the way --cudart static is for CUDA: there
rem is no static HIP runtime to link (CLAUDE.md item 4) -- libamdhip64.dll
rem and its dependencies (amd_comgr, hiprtc, etc., all under C:\rocm\bin)
rem MUST ship alongside bench.exe, or copy them next to it before running.
rem This mirrors the same problem already noted for the planned Linux HIP
rem build ($ORIGIN rpath); on Windows the fix is "ship the DLLs", not a
rem linker flag.
rem BENCH_HIP_BUILD: not used by any CUDA-side code -- boinc_support.cpp
rem reads it (only under HAVE_BOINC) to pick BOINC's "ATI" vendor string
rem instead of "NVIDIA" when checking a client GPU assignment. Defined here
rem unconditionally (harmless when HAVE_BOINC is off) so it doesn't need to
rem be threaded through as a separate knob.
set "CFLAGS=/nologo /O2 /W3 /MT -D_CRT_SECURE_NO_WARNINGS -DBENCH_HIP_BUILD %CF_LMAX_DEF% %DEFS%"
set "CXXFLAGS=/nologo /O2 /W3 /MT /EHsc -D_CRT_SECURE_NO_WARNINGS -DBENCH_HIP_BUILD %CF_LMAX_DEF% %DEFS%"
set "HIPFLAGS=-O2 -std=c++17 %HIP_ARCH% %HIP_DEVLIB% -D_CRT_SECURE_NO_WARNINGS -DBENCH_HIP_BUILD %CF_LMAX_DEF% %DEFS%"

echo Building host C objects with cl.exe... (GFX_ARCH=%GFX_ARCH% CF_LMAX=%CF_LMAX% build=%GIT_DESC%)
for %%F in (fb_load.c verify_cpu.c poly.c primes.c rfb.c fb_cado.c platform.c) do (
    cl %CFLAGS% /std:c11 /c %%F || exit /b 1
)

cl %CFLAGS% /std:c11 -DBENCH_GIT_DESC=\"%GIT_DESC%\" -DBENCH_DEFS=\"%DEFS%\" /c runlog.c || exit /b 1
cl %CFLAGS% /std:c11 -DFBGEN_LIBRARY /c fbgen.c /Fofbgen_lib.obj || exit /b 1
rem Suffixed /Fo, not the bare boinc_support.obj build_windows.bat (CUDA)
rem also compiles from the same source in this same directory: the two
rem builds pass different flags (-DBENCH_HIP_BUILD selects "ATI" over
rem "NVIDIA" in the compiled object), so a bare shared filename means
rem whichever script compiles it last silently wins for BOTH exes the next
rem time either is linked without a fresh recompile -- a real hazard found
rem by code review, not yet exercised by any build order used so far.
cl %CXXFLAGS% /std:c++17 /c boinc_support.cpp /Foboinc_support_hip.obj || exit /b 1

echo Building HIP objects with hipcc (cl.exe as host compiler)...
hipcc %HIPFLAGS% -c bench_main_hip.cpp -o bench_main_hip.obj || exit /b 1
hipcc %HIPFLAGS% -c bench_kernels.hip -o bench_kernels_hip.obj || exit /b 1
hipcc %HIPFLAGS% -DFBGEN_GPU_LIBRARY -c fbgen_gpu.hip -o fbgen_gpu_hip_lib.obj || exit /b 1

echo Linking bench_hip.exe...
hipcc %HIP_ARCH% %HIP_DEVLIB% -o bench_hip.exe ^
    bench_main_hip.obj bench_kernels_hip.obj fbgen_gpu_hip_lib.obj fb_load.obj verify_cpu.obj poly.obj ^
    primes.obj rfb.obj fb_cado.obj platform.obj runlog.obj fbgen_lib.obj ^
    boinc_support_hip.obj || exit /b 1

echo Built %CD%\bench_hip.exe (GFX_ARCH=%GFX_ARCH% CF_LMAX=%CF_LMAX% build=%GIT_DESC%)
echo NOTE: bench_hip.exe needs C:\rocm\bin on PATH (libamdhip64.dll etc.) to
echo       run -- there is no static HIP runtime to link it against. Named
echo       bench_hip.exe (not bench.exe) deliberately, so this never
echo       collides with a bench.exe built by build_windows.bat (CUDA) in
echo       the same directory.
exit /b 0

:do_clean
rem Deliberately does NOT touch the shared host C objects (fb_load.obj,
rem platform.obj, etc.) or bench.exe -- those belong to build_windows.bat's
rem CUDA build too (identical source, identical cl.exe flags, so they're
rem safely shared/reused either way) and this script must not clobber that
rem build's own clean state.
del /q *_hip.obj *_hip_lib.obj bench_hip.exe 2>nul
echo Cleaned Windows HIP build products in %CD%
exit /b 0
