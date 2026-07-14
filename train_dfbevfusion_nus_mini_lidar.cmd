@echo off
setlocal
call "C:\Users\player\miniconda3\condabin\conda.bat" activate mm3d
set "PYTHONPATH=E:\mmdetection3d"
python tools\train.py projects\DFBEVFusion\configs\bevfusion_lidar_voxel03_second_secfpn_8xb4-cyclic-20e_nus-3d.py --work-dir work_dirs\dfbevfusion_nus_mini_lidar --amp
endlocal
