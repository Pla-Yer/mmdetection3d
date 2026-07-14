@echo off
setlocal
if "%~1"=="" (
  echo Usage: train_bevfusion_nus_mini_fusion.cmd ^<lidar_ckpt^> ^<swin_pretrain_ckpt^>
  exit /b 1
)
if "%~2"=="" (
  echo Usage: train_bevfusion_nus_mini_fusion.cmd ^<lidar_ckpt^> ^<swin_pretrain_ckpt^>
  exit /b 1
)
call "C:\Users\player\miniconda3\condabin\conda.bat" activate mm3d
set "PYTHONPATH=E:\mmdetection3d"
python tools\train.py projects\BEVFusion\configs\bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-6e_nus-mini-3d.py --work-dir work_dirs\bevfusion_nus_mini_fusion --amp --cfg-options load_from="%~1" model.img_backbone.init_cfg.checkpoint="%~2"
endlocal
