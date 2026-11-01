# DATASCAPE 2026: E-Commerce Package Quality Control

🥈 **2nd place** solution for the [DATASCAPE 2026 Data Mining Competition](https://www.kaggle.com/competitions/datascape-2026-data-mining-competition/) on Kaggle, hosted by HIMA Data Science, Telkom University.

| Metric | Score |
|---|---|
| Public LB (Macro F1) | **0.96532** |
| Private LB (Macro F1) | **0.92663** |
| 5-fold OOF (Macro F1) | 0.9352 |

## Task

Classify photos of shipping packages into three classes, scored by Macro F1:

| Class | Meaning | Train images |
|---|---|---|
| `Aman` | Intact, undamaged box | 8,800 |
| `Rusak` | Physically damaged: dented, torn, wet, crushed | 3,001 |
| `Lainnya` | Non-standard or out-of-distribution packaging: food packs, plastic bags, paper bags | 423 |

The dataset has 12,224 labeled training images and 3,057 test images. It is heavily imbalanced, comes from mixed sources (catalog shots, courier photos, renders, offline augmentations), and the labels are noisy: 26 byte-identical images carry conflicting labels across classes.

## Approach

**1. Data cleaning** (12,224 → 11,299 training images)
- **MD5 conflicts:** drop byte-identical images whose labels disagree.
- **Source profiling:** unify labels within image sources whose content is clearly one class.
- **Confident-learning pruning:** a linear probe on frozen DINOv3 features flags images whose labels it consistently rejects out-of-fold. 246 were removed, with a per-class threshold and a cap on how many can be dropped.

**2. Model:** [DINOv3 ViT-7B/16](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-lvd1689m) (`facebook/dinov3-vit7b16-pretrain-lvd1689m`)
- Blocks 0–33 frozen in bf16. The last 6 blocks and the final norm are fine-tuned in fp32 after a 1-epoch frozen warm-up.
- Features: `[CLS ‖ mean patch tokens]` (8192-d) → LayerNorm → multi-sample dropout linear head (5 samples).
- Focal loss (γ = 2) with label smoothing (0.02), a class-balanced sampler, layer-wise LR decay (0.85), and EMA weights (0.999).
- Input size 320×320 with ImageNet normalization.

**3. Validation and final training**
- `StratifiedGroupKFold` with 5 folds × 10 epochs. Exact and near-duplicates, including rotated and mirrored copies found via dHash, pHash, and embedding similarity, share one group so nothing leaks between folds.
- The final model is retrained on 100% of the cleaned data for 9 epochs (the median best epoch across folds).
- Inference uses horizontal-flip TTA.

| Configuration | Train images | F1 Aman | F1 Rusak | F1 Lainnya | OOF Macro F1 | Public / Private |
|---|---|---|---|---|---|---|
| Linear probe, frozen backbone | 11,545 | 0.908 | 0.683 | 0.600 | 0.7303 | not submitted |
| **6 blocks unfrozen + cleaned labels** | **11,299** | **0.983** | **0.930** | **0.892** | **0.9352** | **0.96532 / 0.92663** |

## Repository Structure

```
.
├── dinov3_vit7b16_cleaned.ipynb   # Full pipeline: EDA → cleaning → CV → retrain → submission
└── docs/
    ├── context.md                 # Competition brief
    └── slides.pdf                 # Final presentation
```

## Reproducing

**Requirements:** one GPU with about 96 GB of VRAM (developed on an RTX 6000 96GB) and PyTorch with CUDA. The notebook installs the remaining dependencies itself. You also need Hugging Face access to the gated DINOv3 weights.

1. **Data.** By default the notebook downloads the data with `kagglehub`, which needs Kaggle API credentials. To use a local copy instead, set `LOCAL_DATA_DIR` to a folder containing `Train_Set/` and `Test_Set/`.
2. **Full training** (about 8 hours). Set `Settings.INFERENCE = False` and run all cells.
3. **Inference only.** Keep `Settings.INFERENCE = True`. The notebook downloads the trained checkpoint (3.75 GB) from `Settings.CHECKPOINT_URL`, or loads it from `Settings.CHECKPOINT_DIR`.

Run outputs (checkpoint, cleaning log, figures, `submission.csv`) are written to `output/dinov3_vit7b16_cleaned/`.

## Team

- Riantama Putra (Leader)
- Farrel Athalla Putra
- Ayu Nagita Cahyaning Wulan
