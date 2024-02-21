#!/bin/bash

# 스크립트 실행 권한 부여: chmod +x script.sh

# 각 설정에 대해 스크립트 실행
# python main2.py --LR 5e-5 --epochs 50 --batch_size 128 --LR_MAX 3e-5 --shuffle
# python main2.py --LR 1e-5 --epochs 30 --batch_size 64 --LR_MAX 6e-6 --shuffle
# python main2.py --LR 1e-6 --epochs 40 --batch_size 32 --LR_MAX 1e-5

# 더 많은 실행 명령을 추가할 수 있습니다.

# Hydra를 사용하여 설정 오버라이드
# python main2.py training.LR=6e-7 training.LR_MAX=3e-5 lora.r=4
# python main2.py training.LR=7e-7 training.LR_MAX=1e-5 lora.r=4
# python main2.py training.LR=7e-7 training.LR_MAX=6e-6 lora.r=4

# python main2.py training.LR=6e-7 training.LR_MAX=3e-5 lora.r=8
python main2.py training.LR=7e-7 training.LR_MAX=1e-5 lora.r=8
# python main2.py training.LR=7e-7 training.LR_MAX=6e-6 lora.r=8

# python main2.py training.LR=6e-7 training.LR_MAX=3e-5 lora.r=16
python main2.py training.LR=7e-7 training.LR_MAX=1e-5 lora.r=16
# python main2.py training.LR=7e-7 training.LR_MAX=6e-6 lora.r=16


# python main2.py training.LR=6e-7 training.LR_MAX=3e-5 lora.r=32
python main2.py training.LR=7e-7 training.LR_MAX=1e-5 lora.r=32
# python main2.py training.LR=7e-7 training.LR_MAX=6e-6 lora.r=32

# python main2.py training.LR=6e-7 training.LR_MAX=3e-5 lora.r=64
# python main2.py training.LR=7e-7 training.LR_MAX=1e-5 lora.r=64
# python main2.py training.LR=7e-7 training.LR_MAX=6e-6 lora.r=64


# python main2.py training.LR=6e-7 training.LR_MAX=3e-5 lora.r=128
# python main2.py training.LR=7e-7 training.LR_MAX=1e-5 lora.r=128
# python main2.py training.LR=7e-7 training.LR_MAX=6e-6 lora.r=128

# python main2.py training.LR=6e-7 training.LR_MAX=3e-5 lora.r=256
# python main2.py training.LR=7e-7 training.LR_MAX=1e-5 lora.r=256
# python main2.py training.LR=7e-7 training.LR_MAX=6e-6 lora.r=256