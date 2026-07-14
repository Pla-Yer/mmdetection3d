@echo off
call "E:\VisualStudio\IDE\Common7\Tools\VsDevCmd.bat" -arch=x64
call "C:\Users\player\miniconda3\condabin\conda.bat" activate mm3d
set "CUDA_HOME=C:\Users\player\miniconda3\envs\mm3d"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set DISTUTILS_USE_SDK=1
python projects\BEVFusion\setup.py develop --no-deps
