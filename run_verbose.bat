@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
nvcc -allow-unsupported-compiler -Xptxas=-v "%~1" -o "%~dpn1.exe" 2>&1
