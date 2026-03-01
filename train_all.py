#!/usr/bin/env python3
"""
一键训练脚本 - 支持训练所有阶段或指定阶段

用法:
    python train_all.py                          # 训练两个阶段 (L -> R-pre)
    python train_all.py --stage L                # 只训练 Stage-L
    python train_all.py --stage R-pre            # 只训练 Stage-R-pre (需要illum_ckpt.pth)
    python train_all.py --stage L,R-pre          # 训练 Stage-L 和 Stage-R-pre
    # Stage-R-adapt 暂时舍弃（deprecated）
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_checkpoint(ckpt_path: str, stage_name: str) -> bool:
    """检查checkpoint是否存在"""
    if Path(ckpt_path).exists():
        print(f"✓ {stage_name} checkpoint found: {ckpt_path}")
        return True
    else:
        print(f"✗ {stage_name} checkpoint NOT found: {ckpt_path}")
        return False


def run_stage(stage: str, config_path: str) -> bool:
    """运行单个训练阶段"""
    stage_name_map = {
        'L': 'Stage-L (Illumination Pretraining)',
        'R-pre': 'Stage-R-pre (Reflectance Pretraining)',
        # 'R-adapt': 'Stage-R-adapt (Domain Adaptation)',  # Deprecated
    }
    stage_script_map = {
        'L': 'train_stage_L.py',
        'R-pre': 'train_stage_R_pre.py',
        # 'R-adapt': 'train_stage_R_adapt.py',  # Deprecated
    }
    
    stage_display_name = stage_name_map.get(stage, stage)
    script_name = stage_script_map.get(stage)
    
    print("\n" + "="*70)
    print(f"{'▶ STARTING':^70}")
    print(f"{stage_display_name:^70}")
    print("="*70)
    
    cmd = ['python', script_name, '--config', config_path]
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode == 0:
        print("\n" + "="*70)
        print(f"{'✓ COMPLETED':^70}")
        print(f"{stage_display_name:^70}")
        print("="*70)
        return True
    else:
        print("\n" + "="*70)
        print(f"{'✗ FAILED':^70}")
        print(f"{stage_display_name:^70}")
        print("="*70)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='一键训练脚本 - 支持全部或分别训练三个阶段',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python train_all.py                          # 训练两个阶段 (L -> R-pre)
  python train_all.py --stage L                # 只训练 Stage-L
  python train_all.py --stage R-pre            # 只训练 Stage-R-pre
  python train_all.py --stage L,R-pre          # 训练 Stage-L 和 Stage-R-pre
    # Stage-R-adapt 暂时舍弃（deprecated）
        """
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='all',
        help='要训练的阶段: L, R-pre, 或用逗号分隔的组合 (默认: all)',
    )
    parser.add_argument(
        '--config-L',
        type=str,
        default='configs/stage_L.yaml',
        help='Stage-L config文件路径',
    )
    parser.add_argument(
        '--config-R-pre',
        type=str,
        default='configs/stage_R_pre.yaml',
        help='Stage-R-pre config文件路径',
    )
    # Stage-R-adapt 暂时舍弃（deprecated）
    args = parser.parse_args()
    
    # 解析要训练的阶段
    if args.stage.lower() == 'all':
        stages_to_train = ['L', 'R-pre']
    else:
        stages_to_train = [s.strip() for s in args.stage.split(',')]
    
    # 验证输入的阶段名称
    valid_stages = {'L', 'R-pre'}
    for stage in stages_to_train:
        if stage not in valid_stages:
            print(f"错误: 无效的阶段 '{stage}'。有效阶段: {valid_stages}")
            sys.exit(1)
    
    print("\n" + "="*70)
    print(f"{'训练计划':^70}")
    print("="*70)
    print(f"阶段: {', '.join(stages_to_train)}")
    print(f"checkpoints目录: {Path('checkpoints').absolute()}")
    print("="*70)
    
    # 检查必要的checkpoint
    checkpoints_dir = Path('checkpoints')
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n检查checkpoint依赖...")
    
    if 'R-pre' in stages_to_train or 'R-adapt' in stages_to_train:
        illum_ckpt = 'checkpoints/illum_ckpt.pth'
        if 'L' not in stages_to_train:
            if not check_checkpoint(illum_ckpt, 'Illumination'):
                print(f"\n警告: Stage-L 不在训练计划中，但 {illum_ckpt} 不存在！")
                print("请先运行 Stage-L 或提供 illum_ckpt.pth")
                sys.exit(1)
    
    
    # 训练阶段
    config_map = {
        'L': args.config_L,
        'R-pre': args.config_R_pre,
        # 'R-adapt': args.config_R_adapt,  # Deprecated
    }
    
    failed_stages = []
    
    for stage in stages_to_train:
        config_path = config_map[stage]
        if not Path(config_path).exists():
            print(f"错误: config文件不存在: {config_path}")
            sys.exit(1)
        
        success = run_stage(stage, config_path)
        if not success:
            failed_stages.append(stage)
    
    # 总结
    print("\n" + "="*70)
    print(f"{'训练总结':^70}")
    print("="*70)
    print(f"总阶段数: {len(stages_to_train)}")
    print(f"成功阶段: {len(stages_to_train) - len(failed_stages)}")
    
    if failed_stages:
        print(f"失败阶段: {', '.join(failed_stages)}")
        print("\n✗ 训练过程中出现失败")
        sys.exit(1)
    else:
        print("\n✓ 所有阶段训练完成!")
        print("\n下一步:")
        print("  python infer.py --mode zero_shot --config configs/infer.yaml")
        # adapt 暂时舍弃（deprecated）
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
