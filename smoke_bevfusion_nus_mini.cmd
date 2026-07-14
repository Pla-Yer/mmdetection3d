@echo off
setlocal
call "C:\Users\player\miniconda3\condabin\conda.bat" activate mm3d
set "PYTHONPATH=E:\mmdetection3d"
python tools\train.py projects\BEVFusion\configs\bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-1x1-1iter_nus-mini-3d.py --work-dir work_dirs\bevfusion_nus_mini_smoke
endlocal
