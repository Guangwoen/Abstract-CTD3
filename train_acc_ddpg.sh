#!/bin/bash

# echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/acc/train/train_acc(ddpg).py" --gpu=1

# echo "ddpg训练完成"

# echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/acc/train/train_acc(ddpg_risk).py" --gpu=1

# echo "ddpg-risk训练完成"

echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/acc/train/train_acc(ddpg_with_model).py" --gpu=1

echo "ddpg-with-model训练完成"

# echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/acc/train/train_acc(ddpg_risk_with_model).py" --gpu=1

# echo "ddpg-risk-with-model训练完成"

echo "所有模型训练完成"
