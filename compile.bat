@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

if "%~1"=="" (
    echo Usage:
    echo   compile.bat path\to\file.cu              - compile and run
    echo   compile.bat path\to\file.cu verbose      - compile with register/spill info
    echo   compile.bat path\to\file.cu banks        - profile shared memory bank conflicts
    echo   compile.bat path\to\file.cu divergence   - profile warp divergence and efficiency
    echo   compile.bat path\to\file.cu profile      - full Nsight Compute profile ^(opens .ncu-rep^)
    exit /b 1
)

set FULLPATH=%~1
set BASENAME=%~n1
set OUTDIR=%~dp1
set OUTPUT=%OUTDIR%%BASENAME%.exe

if "%~2"=="verbose" (
    nvcc -allow-unsupported-compiler -Xptxas=-v "%FULLPATH%" -o "%OUTPUT%"
) else if "%~2"=="profile" (
    nvcc -allow-unsupported-compiler -lineinfo "%FULLPATH%" -o "%OUTPUT%"
) else (
    nvcc -allow-unsupported-compiler "%FULLPATH%" -o "%OUTPUT%"
)
if not %errorlevel%==0 (
    echo.
    echo Compilation failed!
    exit /b 1
)

echo.
echo Compiled successfully: %OUTPUT%

if "%~2"=="banks" (
    echo Profiling shared memory bank conflicts with Nsight Compute...
    echo.
    powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/k ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \"%OUTPUT%\" ^& pause'"
) else if "%~2"=="divergence" (
    echo Profiling warp divergence and efficiency with Nsight Compute...
    echo.
    powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/k ncu --metrics smsp__thread_inst_executed_per_inst_executed.ratio,smsp__warps_active.avg.pct_of_peak_sustained_active,smsp__sass_branch_targets_threads_divergent.sum,smsp__sass_branch_targets_threads_divergent.avg.pct_of_peak_sustained_elapsed \"%OUTPUT%\" ^& pause'"
) else if "%~2"=="profile" (
    echo Running full Nsight Compute analysis...
    echo.
    powershell -Command "Start-Process cmd -Verb RunAs -ArgumentList '/k ncu --set full -o \"%OUTDIR%%BASENAME%-analysis\" \"%OUTPUT%\" ^& echo Saved to %OUTDIR%%BASENAME%-analysis.ncu-rep ^& pause'"
) else (
    echo Running...
    echo.
    "%OUTPUT%"
)
