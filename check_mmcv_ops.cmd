@echo off
call "E:\VisualStudio\IDE\Common7\Tools\VsDevCmd.bat" -arch=x64
call "C:\Users\player\miniconda3\condabin\conda.bat" activate mm3d
set "CUDA_HOME=C:\Users\player\miniconda3\envs\mm3d"
set "PATH=%CUDA_HOME%\bin;%CUDA_HOME%\Library\bin;%CUDA_HOME%\Lib\site-packages\torch\lib;%PATH%"
python -c "from mmcv.ops import nms; print('mmcv ops ok')"
