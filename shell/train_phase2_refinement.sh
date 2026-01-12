#!/bin/bash
# =====================================================
# Phase 2: Texture Refinement Training Script
# =====================================================
# IMPORTANT: Run Phase 1 (LBBDM-Hierarchical) first until convergence,
# then run this script to train Phase 2 refinement module.
#
# Prerequisites:
# 1. Phase 1 model trained and saved at:
#    results/dataset_name/LBBDM-Hierarchical-oneHot_boundary_add_edge/checkpoint/last_model.pth
# 2. Update phase1_model_path in configs/Template-Refinement.yaml if needed
# =====================================================

# =====================================================
# Configuration
# =====================================================
CONFIG_FILE="configs/Template-Refinement.yaml"
GPU_ID="0"

# =====================================================
# Train Phase 2 Refinement Module
# =====================================================
echo "=========================================="
echo "Phase 2: Training Texture Refinement Module"
echo "Config: ${CONFIG_FILE}"
echo "GPU: ${GPU_ID}"
echo "=========================================="

python3 main.py \
    --config ${CONFIG_FILE} \
    --train \
    --sample_at_start \
    --save_top \
    --gpu_ids ${GPU_ID}

echo "=========================================="
echo "Phase 2 Training Complete!"
echo "=========================================="

# =====================================================
# Optional: Evaluation
# =====================================================
# Uncomment to run evaluation after training:
#
# python3 main.py \
#     --config ${CONFIG_FILE} \
#     --sample_to_eval \
#     --gpu_ids ${GPU_ID}
