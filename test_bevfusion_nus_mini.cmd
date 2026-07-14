@echo off
setlocal
if "%~1"=="" (
  echo Usage: test_bevfusion_nus_mini.cmd ^<checkpoint^> [lidar^|fusion]
  exit /b 1
)
set "CFG=projects\BEVFusion\configs\bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-6e_nus-mini-3d.py"
if /I "%~2"=="lidar" set "CFG=projects\BEVFusion\configs\bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-6e_nus-mini-3d.py"
call "C:\Users\player\miniconda3\condabin\conda.bat" activate mm3d
set "PYTHONPATH=E:\mmdetection3d"
python tools\test.py %CFG% "%~1"
endlocal
