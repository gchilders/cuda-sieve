@echo off
setlocal EnableExtensions
rem ---------------------------------------------------------------------------
rem wintest.bat -- cross-platform output check for the native Windows build.
rem
rem Runs ONE small fixed band and reports four numbers to compare against a
rem Linux run of the identical command. This is deliberately not testsieve.sh:
rem that harness measures yield and timing, which are machine-specific and do
rem not compare across platforms. What needs checking here is that Windows
rem writes the SAME BYTES, which is what the binary-mode fix was for.
rem
rem The decisive number is "CR bytes", which MUST be 0. A text-mode build would
rem write \r\n and put one CR on every line. That number cannot be confounded
rem by floating-point differences between compilers, which the hash can.
rem
rem Needs in this directory: bench.exe, input.job, fbase.m16
rem ---------------------------------------------------------------------------

if not exist bench.exe  ( echo error: bench.exe not here -- run build_windows.bat first & exit /b 1 )
if not exist input.job  ( echo error: input.job not here -- copy it from the Linux tree & exit /b 1 )
if not exist fbase.m16  ( echo error: fbase.m16 not here -- copy it from the Linux tree, 131 MB & exit /b 1 )

rem Wildcard: this clears wintest.dat plus its .part, .part.ckpt and .lock
rem sidecars. A leftover .part would make the run RESUME rather than start
rem fresh -- a different code path, and a different answer.
del /q wintest.dat* wintest.out 2>nul

echo Running one band: q from 50000000 to 50001999, logI 16, J 16384.
echo About 90 special-q -- well under a minute on a modern card.
bench.exe --pipeline --cofactor --poly input.job --fb1 fbase.m16 ^
    --sq-side 1 --logI 16 --J 16384 --qrange 50000000:50001999 ^
    --relations wintest.dat --log wintest.runlog --log-every 1 > wintest.out 2>&1
if errorlevel 1 (
    echo.
    echo BAND FAILED -- matching lines follow, full output in wintest.out:
    findstr /I /C:"error" /C:"cannot" /C:"refus" /C:"does not fit" wintest.out
    exit /b 1
)

if not exist wintest.dat (
    echo.
    echo BAND REPORTED SUCCESS BUT WROTE NO RELATION FILE.
    echo That is itself a finding: the .part rename did not happen. See wintest.out.
    exit /b 1
)

echo.
echo ============ compare these against the Linux reference ============
powershell -NoProfile -Command "$b=[IO.File]::ReadAllBytes('wintest.dat'); $cr=@($b -eq 13).Count; $lf=@($b -eq 10).Count; $h=(Get-FileHash wintest.dat -Algorithm SHA256).Hash.ToLower(); Write-Output ('relations (LF count): ' + $lf); Write-Output ('bytes:                ' + $b.Length); Write-Output ('CR bytes:             ' + $cr + '   <-- MUST be 0'); Write-Output ('SHA-256:              ' + $h)"
echo ===================================================================
echo.
echo Linux reference for the identical command:
echo   relations (LF count): 5574
echo   bytes:                750113
echo   CR bytes:             0
echo   SHA-256:              c34ee2e570b49f18923056331105b212987c0db6c3a2408debbe755f579c4988
echo.
echo CR bytes must be 0. If the other three match too, the Windows build
echo produces byte-identical relations. Send the block above either way.
exit /b 0
