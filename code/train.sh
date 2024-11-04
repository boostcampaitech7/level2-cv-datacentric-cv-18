#!/bin/bash

# 데이터 디렉토리 및 로그 디렉토리 설정
DATA_DIR="/data/ephemeral/home/level2-cv-datacentric-cv-18/data"
LOG_DIR="/data/ephemeral/home/level2-cv-datacentric-cv-18/nohup_output"

# 증강 방법 배열
aug_methods=("gaussian" "glass" "motion" "median" "advanced" "blur" "gaussnoise" "isonoise" "multiplicativenoise" 
             "imagecompression" "jpegcompression" "randombrightness" "randomcontrast" "randombrightnesscontrast" 
             "superpixels" "randomfog" "randomrain" "randomshadow" "randomsnow" "randomsunflare" 
             "clahe" "emboss" "randomtonecurve" "downscale" "equalize" "fancypca")

# 증강 방법에 대해 순차적으로 실행
for method in "${aug_methods[@]}"; do
    echo "Running augmentation method: $method"
    
    # nohup으로 실행하고 에러가 발생하면 스크립트를 종료
    if nohup python3 train.py --data_dir "$DATA_DIR" --aug_method "$method" > "$LOG_DIR/$method.log" 2>&1; then
        echo "$method completed successfully."
    else
        echo "Error occurred while running $method. Check log file: $LOG_DIR/$method.log"
        exit 1  # 스크립트 종료
    fi
done