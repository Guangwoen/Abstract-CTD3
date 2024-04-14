#!/bin/bash

# echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/train/train_lanekeeping(td3).py" --gpu=1

# echo "td3训练完成"

# echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/train/train_lanekeeping(td3_with_model).py" --gpu=1

# echo "td3-with-model训练完成"

# echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/train/train_lanekeeping(td3_risk).py" --gpu=1

# echo "td3-risk训练完成"

echo "y" | python "/home/akihi/data1/Abstract-CTD3-main-master/lanekeeping/train/train_lanekeeping(td3_risk_with_model).py" --gpu=1

echo "td3-risk-with-model训练完成"

echo "所有模型训练完成"
