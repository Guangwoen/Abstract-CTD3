#!/bin/bash

# echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/train/train_lanekeeping(ddpg).py" --gpu=1

# echo "ddpg训练完成"

# echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/train/train_lanekeeping(ddpg_with_model).py" --gpu=1

# echo "ddpg-with-model训练完成"

echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/train/train_lanekeeping(ddpg_risk).py" --gpu=0

echo "ddpg-risk训练完成"

# echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/train/train_lanekeeping(ddpg_risk_with_model).py" --gpu=1

# echo "ddpg-risk-with-model训练完成"

echo "所有模型训练完成"
