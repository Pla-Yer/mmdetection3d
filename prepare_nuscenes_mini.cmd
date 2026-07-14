@echo off
setlocal
call "C:\Users\player\miniconda3\condabin\conda.bat" activate mm3d
set "PYTHONPATH=E:\mmdetection3d"
python tools\create_data.py nuscenes --root-path .\data\nuscenes --out-dir .\data\nuscenes --version v1.0-mini --extra-tag nuscenes_mini
python tools\create_data.py nuscenes --root-path .\data\nuscenes --out-dir .\data\nuscenes --version v1.0-mini --extra-tag nuscenes_mini --only-gt-database
endlocal
