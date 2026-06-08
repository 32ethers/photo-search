"""兼容性补丁 - 必须在 transformers 之前导入"""

import os

# 抑制 transformers 5.x 对旧模型 config 的 token ID 范围误报
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch

# transformers 5.x 引用 torch.float8_e8m0fnu (torch 2.7+)
# cu124 index 最高 torch 2.6，推理不需要 FP8，用 float8_e4m3fn 替代
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.float8_e4m3fn
