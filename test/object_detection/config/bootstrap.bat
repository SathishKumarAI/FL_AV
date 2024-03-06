:: ### don't modify this part ###
:: ##############################


:: ### please customize your script in this region ####
@echo off

:: Activate Conda environment
call conda activate federated_learning

:: Install Python packages
pip install opencv-python-headless pandas matplotlib seaborn addict

:: Set DATA_PATH
set DATA_PATH=%userprofile%\fedcv_data

:: Check if DATA_PATH exists
if exist %DATA_PATH% (
    echo Exist %DATA_PATH%
) else (
    :: Create DATA_PATH directory
    mkdir %DATA_PATH%

    :: Run download_coco128.bat script
    set cur_dir=%cd%
    call %cur_dir%\..\data\coco128\download_coco128.bat
)


:: ### don't modify this part ###
echo [FedML]Bootstrap Finished
:: ##############################