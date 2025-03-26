# aes_kit/core/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# 標準的な損失関数はそのまま利用可能
MSELoss = nn.MSELoss
L1Loss = nn.L1Loss

# QWK損失 (参考: 実装は複雑で、性能が安定しない場合もある)
# class QWKLoss(nn.Module):
#     def __init__(self, min_rating, max_rating):
#         super().__init__()
#         # ... QWK損失の実装 ...
#     def forward(self, input, target):
#         # ...
#         return loss

# MSEで学習し、評価でQWKを使うのが一般的