import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from src.models.illumination import IlluminationNet
from src.models.adarenet import AdaReNet
from src.models.retinex import RetinexAdaReNet

# 构建模型
illumination = IlluminationNet()
adarenet = AdaReNet()
model = RetinexAdaReNet(illumination, adarenet, omega=5.0, tau=1e-3, eps=1e-6)  # omega=5.0
model.cuda()

# 加载检查点 - 正确方式
model.illumination.load_state_dict(torch.load('checkpoints/illum_ckpt.pth', map_location='cuda'))
model.adarenet.load_state_dict(torch.load('checkpoints/denoise_adapt_ckpt.pth', map_location='cuda'))
model.eval()

# 加载测试图
low = np.array(Image.open('/home/yannayanna/datasets/LOLv2_raw/LOL-v2/Real_captured/Test/Low/low00690.png').convert('RGB')).astype(np.float32) / 255.0
low_tensor = torch.from_numpy(low).permute(2, 0, 1).unsqueeze(0).cuda()

# 加载高光参考
high = np.array(Image.open('/home/yannayanna/datasets/LOLv2_raw/LOL-v2/Real_captured/Test/Normal/normal00690.png').convert('RGB')).astype(np.float32) / 255.0

with torch.no_grad():
    L_T, L_e = model.compute_illumination(low_tensor)
    P_ref = model.compute_pref(low_tensor, L_T)
    output = model(low_tensor)
    
    L_T_np = L_T[0].squeeze(0).cpu().numpy() if L_T.shape[1] == 1 else L_T[0].permute(1, 2, 0).cpu().numpy()
    L_e_np = L_e[0].squeeze(0).cpu().numpy() if L_e.shape[1] == 1 else L_e[0].permute(1, 2, 0).cpu().numpy()
    P_ref_np = P_ref[0].permute(1, 2, 0).cpu().numpy()
    I_hat_np = output['I_hat'][0].permute(1, 2, 0).cpu().numpy()

print('=== 问题诊断 ===')
print()
print(f'输入(Low): {low.mean():.4f}')
print(f'GT(High): {high.mean():.4f}')
print(f'输出(I_hat): {I_hat_np.mean():.4f}')
print()
print(f'L_T均值(1ch): {L_T_np.mean():.4f}')
print(f'L_e均值(1ch): {L_e_np.mean():.4f}')
print()
print('亮度差异:')
print(f'  输出 vs GT: {I_hat_np.mean() - high.mean():.4f} (差值)')
print()
print('局部对比度:')
print(f'  输出std: {I_hat_np.std():.4f}')
print(f'  GT std: {high.std():.4f}')
print()
print('=== 问题分析 ===')
if I_hat_np.mean() < high.mean() - 0.05:
    print('❌ 输出太暗，需要提亮')
    if L_T_np.mean() < 0.1:
        print('   → L_T太小，建议: 增加omega(2.0→3.0-4.0) 或 增大max_pool_kernel(15→31)')
if I_hat_np.std() < high.std() - 0.02:
    print('❌ 清晰度太低，对比度不足')
    print('   → 建议: 增大max_pool_kernel保留更多细节 或 调整delta_lambda')
if I_hat_np.mean() > high.mean() + 0.05:
    print('⚠️  局部过亮/曝光')
    print('   → 建议: 减少omega 或 增大max_pool_kernel')
