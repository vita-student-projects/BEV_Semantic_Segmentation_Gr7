#!/usr/bin/env bash

# Training
bash ./tools/dist_train.sh ./projects/configs/bevformer/segdet_surr.py 0

bash ./tools/dist_train.sh ./projects/configs/bevformer/segdet_front_finetune.py 0

# Test
bash ./tools/dist_test.sh ./projects/configs/bevformer/segdet_surr.py './ckpts/best_segdet_surr.pth' 'full'

bash ./tools/dist_test.sh ./projects/configs/bevformer/segdet_front_finetune.py './ckpts/best_segdet_front.pth' 'half'

