# CVPR 2025, Paper ID: 4428

## 1. Requirements

Python 3.8+, Pytorch 1.9.0, Cuda 11.1, , opencv-python



## 2. Training & Testing

- model training:

    `bash ./tools/dist_train.sh config_file K (GPU number)`
- 
- model evaluation:

    `python tools/test.py config_file checkpoints_file --eval mIoU`
    `python tools/fps_test.py config_file --height height of the test image --width width of the test image`

- Set experiment settings:
     
    browse `config` folder

    