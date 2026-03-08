#!/usr/bin/env python
"""
EnlightenGAN 推理脚本
基于 Unet_resize_conv 架构 (use_norm=1, self_attention=False, skip=1)
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


class Unet_resize_conv(nn.Module):
    """
    重建 EnlightenGAN 的 Unet_resize_conv 网络
    基于预训练权重的 keys 推断参数: use_norm=1, self_attention=True, skip=1, tanh=True
    """
    def __init__(self, skip=1.0, self_attention=True):
        super(Unet_resize_conv, self).__init__()
        
        self.skip = skip
        self.self_attention = self_attention
        p = 1
        
        # Encoder
        # self_attention=True means 4 channels input (RGB + gray)
        input_channels = 4 if self_attention else 3
        self.conv1_1 = nn.Conv2d(input_channels, 32, 3, padding=p)
        self.LReLU1_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU1_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.max_pool1 = nn.MaxPool2d(2)
        
        if self_attention:
            self.downsample_1 = nn.MaxPool2d(2)
            self.downsample_2 = nn.MaxPool2d(2)
            self.downsample_3 = nn.MaxPool2d(2)
            self.downsample_4 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=p)
        self.LReLU2_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU2_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.max_pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=p)
        self.LReLU3_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU3_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.max_pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=p)
        self.LReLU4_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU4_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.max_pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(256, 512, 3, padding=p)
        self.LReLU5_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=p)
        self.LReLU5_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn5_2 = nn.BatchNorm2d(512)

        # Decoder
        self.deconv5 = nn.Conv2d(512, 256, 3, padding=p)
        self.conv6_1 = nn.Conv2d(512, 256, 3, padding=p)
        self.LReLU6_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_1 = nn.BatchNorm2d(256)
        self.conv6_2 = nn.Conv2d(256, 256, 3, padding=p)
        self.LReLU6_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn6_2 = nn.BatchNorm2d(256)

        self.deconv6 = nn.Conv2d(256, 128, 3, padding=p)
        self.conv7_1 = nn.Conv2d(256, 128, 3, padding=p)
        self.LReLU7_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_1 = nn.BatchNorm2d(128)
        self.conv7_2 = nn.Conv2d(128, 128, 3, padding=p)
        self.LReLU7_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn7_2 = nn.BatchNorm2d(128)

        self.deconv7 = nn.Conv2d(128, 64, 3, padding=p)
        self.conv8_1 = nn.Conv2d(128, 64, 3, padding=p)
        self.LReLU8_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_1 = nn.BatchNorm2d(64)
        self.conv8_2 = nn.Conv2d(64, 64, 3, padding=p)
        self.LReLU8_2 = nn.LeakyReLU(0.2, inplace=True)
        self.bn8_2 = nn.BatchNorm2d(64)

        self.deconv8 = nn.Conv2d(64, 32, 3, padding=p)
        self.conv9_1 = nn.Conv2d(64, 32, 3, padding=p)
        self.LReLU9_1 = nn.LeakyReLU(0.2, inplace=True)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.conv9_2 = nn.Conv2d(32, 32, 3, padding=p)
        self.LReLU9_2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv10 = nn.Conv2d(32, 3, 1)
        self.tanh = nn.Tanh()

    def forward(self, input, gray=None):
        """
        Forward pass.
        
        Args:
            input: [B, 3, H, W] in [-1, 1]
            gray: [B, 1, H, W] attention map, optional
        """
        if self.self_attention:
            if gray is None:
                # 计算 gray: 与 EnlightenGAN 数据集中的计算方式一致
                # input 在 [-1, 1] 范围，先转到 [0, 2]，再计算反转灰度
                r, g, b = input[:, 0:1, :, :] + 1, input[:, 1:2, :, :] + 1, input[:, 2:3, :, :] + 1
                gray = 1.0 - (0.299 * r + 0.587 * g + 0.114 * b) / 2.0
            gray_2 = self.downsample_1(gray)
            gray_3 = self.downsample_2(gray_2)
            gray_4 = self.downsample_3(gray_3)
            gray_5 = self.downsample_4(gray_4)
            
            # Encoder with self-attention
            x = self.bn1_1(self.LReLU1_1(self.conv1_1(torch.cat((input, gray), 1))))
        else:
            x = self.bn1_1(self.LReLU1_1(self.conv1_1(input)))
        
        conv1 = self.bn1_2(self.LReLU1_2(self.conv1_2(x)))
        x = self.max_pool1(conv1)

        x = self.bn2_1(self.LReLU2_1(self.conv2_1(x)))
        conv2 = self.bn2_2(self.LReLU2_2(self.conv2_2(x)))
        x = self.max_pool2(conv2)

        x = self.bn3_1(self.LReLU3_1(self.conv3_1(x)))
        conv3 = self.bn3_2(self.LReLU3_2(self.conv3_2(x)))
        x = self.max_pool3(conv3)

        x = self.bn4_1(self.LReLU4_1(self.conv4_1(x)))
        conv4 = self.bn4_2(self.LReLU4_2(self.conv4_2(x)))
        x = self.max_pool4(conv4)

        x = self.bn5_1(self.LReLU5_1(self.conv5_1(x)))
        if self.self_attention:
            x = x * gray_5
        conv5 = self.bn5_2(self.LReLU5_2(self.conv5_2(x)))
        
        # Decoder with self-attention
        conv5 = F.interpolate(conv5, scale_factor=2, mode='bilinear', align_corners=False)
        if self.self_attention:
            conv4 = conv4 * gray_4
        up6 = torch.cat([self.deconv5(conv5), conv4], 1)
        x = self.bn6_1(self.LReLU6_1(self.conv6_1(up6)))
        conv6 = self.bn6_2(self.LReLU6_2(self.conv6_2(x)))

        conv6 = F.interpolate(conv6, scale_factor=2, mode='bilinear', align_corners=False)
        if self.self_attention:
            conv3 = conv3 * gray_3
        up7 = torch.cat([self.deconv6(conv6), conv3], 1)
        x = self.bn7_1(self.LReLU7_1(self.conv7_1(up7)))
        conv7 = self.bn7_2(self.LReLU7_2(self.conv7_2(x)))

        conv7 = F.interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=False)
        if self.self_attention:
            conv2 = conv2 * gray_2
        up8 = torch.cat([self.deconv7(conv7), conv2], 1)
        x = self.bn8_1(self.LReLU8_1(self.conv8_1(up8)))
        conv8 = self.bn8_2(self.LReLU8_2(self.conv8_2(x)))

        conv8 = F.interpolate(conv8, scale_factor=2, mode='bilinear', align_corners=False)
        if self.self_attention:
            conv1 = conv1 * gray
        up9 = torch.cat([self.deconv8(conv8), conv1], 1)
        x = self.bn9_1(self.LReLU9_1(self.conv9_1(up9)))
        conv9 = self.LReLU9_2(self.conv9_2(x))

        latent = self.conv10(conv9)
        latent = self.tanh(latent)
        
        # Skip connection
        output = latent + input * self.skip
        
        return output, latent


def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 16

    if width_org % divide != 0 or height_org % divide != 0:
        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


class EnlightenGANInference:
    def __init__(self, ckpt_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        if ckpt_path is None:
            ckpt_path = "/home/yannayanna/projects/EnlightenGAN/checkpoints/enlightening/200_net_G_A.pth"
        
        print(f"[EnlightenGAN] Loading model from {ckpt_path}")
        
        # 创建与预训练模型匹配的网络 (self_attention=True based on checkpoint shape)
        # skip=0.8 is the default value in EnlightenGAN
        self.model = Unet_resize_conv(skip=0.8, self_attention=True).to(self.device)
        
        # 加载权重
        state_dict = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        # 处理 DataParallel 的 state dict
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 去掉 'module.' 前缀
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        print("[EnlightenGAN] Model loaded successfully")
    
    @torch.no_grad()
    def enhance(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Enhance a single image tensor.
        
        Args:
            img_tensor: [C, H, W] in [0, 1]
            
        Returns:
            Enhanced tensor [C, H, W] in [0, 1]
        """
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # 转换到 [-1, 1] 范围
        img_tensor = img_tensor * 2 - 1
        img_tensor = img_tensor.to(self.device)
        
        # Pad to divisible by 16
        img_padded, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(img_tensor)
        
        # Forward pass
        output, _ = self.model(img_padded)
        
        # Remove padding
        output = pad_tensor_back(output, pad_left, pad_right, pad_top, pad_bottom)
        
        # 转换回 [0, 1] 范围
        output = (output + 1) / 2
        
        return output.squeeze(0).clamp(0, 1)
    
    def process_directory(self, input_dir: str, output_dir: str):
        """Process all images in a directory."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
        image_files = [f for f in image_files if f.is_file()]
        
        print(f"[EnlightenGAN] Processing {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="EnlightenGAN"):
            # Load image
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
            
            # Enhance
            enhanced = self.enhance(img_tensor)
            
            # Save
            enhanced_np = enhanced.cpu().permute(1, 2, 0).numpy()
            enhanced_np = (np.clip(enhanced_np, 0, 1) * 255).astype(np.uint8)
            out_path = output_dir / img_path.name
            Image.fromarray(enhanced_np).save(str(out_path))
        
        print(f"[EnlightenGAN] Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None)
    args = parser.parse_args()
    
    model = EnlightenGANInference(ckpt_path=args.ckpt)
    model.process_directory(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
