@echo off
call "E:\VisualStudio\IDE\Common7\Tools\VsDevCmd.bat" -arch=x64
call "C:\Users\player\miniconda3\condabin\conda.bat" activate mm3d
set "CUDA_HOME=C:\Users\player\miniconda3\envs\mm3d"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\Library\bin;%CUDA_HOME%\Lib\site-packages\torch\lib;%PATH%"
set DISTUTILS_USE_SDK=1
set MMCV_WITH_OPS=1
pip install -v --no-build-isolation --no-binary mmcv mmcv==2.1.0
