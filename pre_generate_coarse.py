"""
Pre-generate Coarse RGB from Phase 1 LBBDM
============================================

This script runs Phase 1 LBBDM inference once on the entire dataset
and saves the coarse RGB outputs to disk. This dramatically speeds up
Phase 2 training (from ~50s/batch to ~0.5s/batch).

Usage:
    python pre_generate_coarse.py --config configs/Template-Hierarchical.yaml --split train
    python pre_generate_coarse.py --config configs/Template-Hierarchical.yaml --split val

NOTE: Use the Phase 1 config (Template-Hierarchical.yaml), NOT Template-Refinement.yaml!

Output Structure:
    {dataset_path}/{split}/coarse_rgb/
        ├── image_001.png  (same name as original in folder B)
        ├── image_002.png
        └── ...
"""

import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import numpy as np

from utils import dict2namespace
from runners.utils import get_dataset
from model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel


def parse_args():
    parser = argparse.ArgumentParser(description='Pre-generate coarse RGB from Phase 1 LBBDM')
    parser.add_argument('-c', '--config', type=str, required=True, 
                        help='Path to Phase 1 config file (Template-Hierarchical.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to Phase 1 checkpoint (overrides config)')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                        help='Dataset split to process')
    parser.add_argument('--batch_size', type=int, default=4, 
                        help='Batch size for inference')
    parser.add_argument('--gpu_id', type=int, default=0, 
                        help='GPU ID to use')
    parser.add_argument('--sample_step', type=int, default=200,
                        help='Number of sampling steps (default 200)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from existing files (skip already generated)')
    return parser.parse_args()


def tensor_to_pil(tensor):
    """Convert tensor [-1, 1] to PIL Image"""
    img = (tensor.cpu().numpy() + 1.0) / 2.0 * 255.0
    img = img.transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def main():
    args = parse_args()
    
    # Load config - MUST use Phase 1 config (Template-Hierarchical.yaml)
    with open(args.config, 'r') as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2namespace(config_dict)
    
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"[PreGenerate] Using device: {device}")
    print(f"[PreGenerate] Config file: {args.config}")
    
    # =====================================================
    # 1. Determine checkpoint path
    # =====================================================
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif hasattr(config.model, 'model_load_path') and config.model.model_load_path:
        checkpoint_path = config.model.model_load_path
    elif hasattr(config.model, 'phase1_model_path') and config.model.phase1_model_path:
        checkpoint_path = config.model.phase1_model_path
    else:
        raise ValueError("No checkpoint path found! Use --checkpoint or set model_load_path in config")
    
    print(f"[PreGenerate] Loading Phase 1 LBBDM from: {checkpoint_path}")
    
    # =====================================================
    # 2. Create model and load weights
    # =====================================================
    phase1_model = LatentBrownianBridgeModel(config.model).to(device)
    phase1_states = torch.load(checkpoint_path, map_location='cpu')
    
    # Check what's in the checkpoint
    print(f"[PreGenerate] Checkpoint keys: {list(phase1_states.keys())}")
    
    if 'model' in phase1_states:
        model_state = phase1_states['model']
    else:
        model_state = phase1_states
    
    # Load with strict=True first to see missing keys
    try:
        phase1_model.load_state_dict(model_state, strict=True)
        print("[PreGenerate] All weights loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"[PreGenerate] WARNING: strict=True failed: {e}")
        print("[PreGenerate] Trying strict=False...")
        missing_keys, unexpected_keys = phase1_model.load_state_dict(model_state, strict=False)
        if missing_keys:
            print(f"[PreGenerate] WARNING: Missing keys: {len(missing_keys)}")
            for k in missing_keys[:10]:
                print(f"    - {k}")
            if len(missing_keys) > 10:
                print(f"    ... and {len(missing_keys) - 10} more")
        if unexpected_keys:
            print(f"[PreGenerate] WARNING: Unexpected keys: {len(unexpected_keys)}")
            for k in unexpected_keys[:10]:
                print(f"    - {k}")
    
    # Handle EMA weights if present
    if 'ema' in phase1_states and phase1_states.get('ema') is not None:
        print("[PreGenerate] Loading EMA weights for better quality...")
        phase1_model.load_state_dict(phase1_states['ema'], strict=False)
    
    phase1_model.eval()
    for param in phase1_model.parameters():
        param.requires_grad = False
    
    print("[PreGenerate] Phase 1 LBBDM loaded successfully")
    
    # =====================================================
    # 3. Load Dataset (use SARtoOptical)
    # =====================================================
    # Override dataset_type to SARtoOptical for pre-generation
    original_dataset_type = getattr(config.data, 'dataset_type', 'SARtoOptical')
    config.data.dataset_type = 'SARtoOptical'
    
    # IMPORTANT: Disable flip for pre-generation to maintain filename alignment
    # Phase 2 training will apply augmentation independently
    config.data.dataset_config.flip = False
    
    print(f"[PreGenerate] Using dataset type: SARtoOptical (original: {original_dataset_type})")
    print(f"[PreGenerate] Flip disabled for pre-generation (filenames will match originals)")
    
    train_dataset, val_dataset, test_dataset = get_dataset(config.data)
    
    if args.split == 'train':
        dataset = train_dataset
    elif args.split == 'val':
        dataset = val_dataset
    else:
        dataset = test_dataset
    
    # Get original filenames from dataset
    # Dataset structure: dataset.imgs_ori.image_paths contains list of file paths
    if hasattr(dataset, 'imgs_ori') and hasattr(dataset.imgs_ori, 'image_paths'):
        original_filenames = [os.path.basename(p) for p in dataset.imgs_ori.image_paths]
        print(f"[PreGenerate] Found {len(original_filenames)} original filenames")
    else:
        original_filenames = None
        print("[PreGenerate] WARNING: Could not get original filenames, using sequential numbering")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Important: keep order for alignment
        num_workers=4,
        drop_last=False
    )
    
    print(f"[PreGenerate] Processing {args.split} split: {len(dataset)} samples")
    
    # =====================================================
    # 3. Create output directory
    # =====================================================
    output_dir = os.path.join(config.data.dataset_config.dataset_path, args.split, 'coarse_rgb')
    os.makedirs(output_dir, exist_ok=True)
    print(f"[PreGenerate] Output directory: {output_dir}")
    
    # Check for existing files if resume mode
    existing_files = set()
    if args.resume:
        existing_files = set(os.listdir(output_dir))
        print(f"[PreGenerate] Resume mode: Found {len(existing_files)} existing files")
    
    # =====================================================
    # 4. Generate and save coarse RGB
    # =====================================================
    sample_idx = 0
    skipped_count = 0
    generated_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating {args.split}"):
            # Determine filenames for this batch first (to check if we can skip)
            batch_size_actual = batch[0].shape[0]
            batch_filenames = []
            batch_needs_generation = False
            
            for i in range(batch_size_actual):
                idx = sample_idx + i
                if original_filenames is not None and idx < len(original_filenames):
                    filename = original_filenames[idx]
                    if not filename.lower().endswith('.png'):
                        filename = os.path.splitext(filename)[0] + '.png'
                else:
                    filename = f'{idx:06d}.png'
                batch_filenames.append(filename)
                
                if filename not in existing_files:
                    batch_needs_generation = True
            
            # Skip entire batch if all files exist (resume mode)
            if args.resume and not batch_needs_generation:
                skipped_count += batch_size_actual
                sample_idx += batch_size_actual
                continue
            
            # Unpack batch: (optical, sar, lc_label, edge_map)
            if len(batch) >= 4:
                x, x_cond_sar, x_cond_lc, x_cond_edge = batch[0], batch[1], batch[2], batch[3]
            elif len(batch) == 3:
                x, x_cond_sar, x_cond_lc = batch[0], batch[1], batch[2]
                x_cond_edge = None
            else:
                x, x_cond_sar = batch[0], batch[1]
                x_cond_lc = None
                x_cond_edge = None
            
            # Move to device
            x_cond_sar = x_cond_sar.to(device)
            
            if x_cond_lc is not None:
                x_cond_lc = x_cond_lc.to(device)
            else:
                x_cond_lc = torch.zeros(x_cond_sar.shape[0], x_cond_sar.shape[2], x_cond_sar.shape[3], 
                                        dtype=torch.long, device=device)
            
            if x_cond_edge is not None:
                x_cond_edge = x_cond_edge.to(device)
            else:
                x_cond_edge = torch.zeros(x_cond_sar.shape[0], 1, x_cond_sar.shape[2], x_cond_sar.shape[3],
                                          device=device)
            
            # Debug: Print first batch info
            if generated_count == 0 and not args.resume:
                print(f"[DEBUG] SAR shape: {x_cond_sar.shape}, range: [{x_cond_sar.min():.3f}, {x_cond_sar.max():.3f}]")
                print(f"[DEBUG] LC shape: {x_cond_lc.shape}, unique values: {x_cond_lc.unique().tolist()[:10]}...")
                print(f"[DEBUG] Edge shape: {x_cond_edge.shape}, range: [{x_cond_edge.min():.3f}, {x_cond_edge.max():.3f}]")
            
            # Generate coarse RGB from Phase 1
            # Use clip_denoised=False to match training config (Template-Hierarchical.yaml has clip_denoised: False)
            clip_denoised = getattr(config.testing, 'clip_denoised', False)
            coarse_rgb = phase1_model.sample(
                x_cond_sar, x_cond_lc, x_cond_edge,
                clip_denoised=clip_denoised
            )
            
            # Debug: Print output info for first generated batch
            if generated_count == 0:
                print(f"[DEBUG] Output coarse_rgb shape: {coarse_rgb.shape}")
                print(f"[DEBUG] Output range: [{coarse_rgb.min():.3f}, {coarse_rgb.max():.3f}]")
            
            # Save each image with original filename
            batch_size = coarse_rgb.shape[0]
            for i in range(batch_size):
                filename = batch_filenames[i]
                
                # Skip if file already exists (resume mode)
                if args.resume and filename in existing_files:
                    skipped_count += 1
                    sample_idx += 1
                    continue
                
                img = tensor_to_pil(coarse_rgb[i])
                img.save(os.path.join(output_dir, filename))
                generated_count += 1
                sample_idx += 1
    
    print(f"[PreGenerate] Done! Generated {generated_count} new images, skipped {skipped_count} existing")
    print(f"[PreGenerate] Total in {output_dir}: {len(os.listdir(output_dir))} files")
    print(f"[PreGenerate] Now update your Phase 2 config to use dataset_type: 'Phase2Refinement'")


if __name__ == '__main__':
    main()
