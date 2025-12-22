# IPEO-SwissIMAGE-MMLearning

Multimodal deep learning project for ENV-540 (Image Processing for Earth Observation), focusing on topic 5: **Mapping Swiss ecosystems from aerial images and environmental variables**. The goal is to predict 17 EUNIS ecosystem classes by combining swissIMAGE RGB orthophotos (50 cm, 100x100 m chips) with 48 standardized SWECO25 environmental variables.

## Project Goals
- Build a training pipeline with an appropriate backbone (pretrained CNN/ViT), loss, and tuned hyperparameters.
- Compare modalities: RGB only, tabular only, and fusion (early/late/intermediate).
- Perform ablations by SWECO thematic groups (e.g., land cover, bioclimate, geology, hydrology, vegetation, population) to interpret variable contribution.
- Evaluate on the provided geographic train/val/test split; report accuracy, macro/micro F1, and per-class metrics; discuss calibration and confusion patterns.
- Visualize results with maps/figures to interpret strengths and failure cases.

## Data
- Samples: 16,925 locations, split geographically into train 60%, val 10%, test 30%.
- Inputs:
  - RGB aerial image: 100x100 m at 50 cm resolution (swissIMAGE), 3 bands.
  - SWECO25 tabular variables: 48 normalized numerical features across 6 themes.
- Target: 17-class EUNIS ecosystem label.
- Access: download from the course-provided link (see ProjectDescription.pdf). Place the raw data under `data/raw/` (suggested) and keep the geographic split intact.

Suggested directory layout (adapt as needed):
```
data/
  raw/
    images/           # swissIMAGE tiles
    sweco/            # tabular CSV/Parquet
    splits/           # train/val/test indices
  processed/          # cached tensors/patches after preprocessing
src/
  data/               # dataloaders, augmentations
  models/             # CNN/ViT backbones, MLPs, fusion heads
  training/           # loops, callbacks, losses
  eval/               # metrics, confusion, calibration
notebooks/            # EDA and quick experiments
configs/              # hyperparams and experiment configs (if using Hydra/JSON/YAML)
```

## Environment Setup
- Python >= 3.10 recommended.
- Core packages: `torch`, `torchvision` or `timm`, `numpy`, `pandas`, `scikit-learn`, `albumentations` or `torchvision.transforms`, `rasterio`, `geopandas`, `matplotlib`, `seaborn`, `tqdm`.
- Optional: `pytorch-lightning` or `lightning`, `hydra-core`, `wandb`/`tensorboard`.

Example setup:
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or cuda wheel
pip install timm numpy pandas scikit-learn albumentations rasterio geopandas matplotlib seaborn tqdm
```

## Recommended Workflow
1) **EDA**: Inspect class distribution, spatial split, and variable ranges (notebooks).
2) **Preprocessing**:
   - Images: resize/crop to a consistent tensor size; apply augmentations; normalize per-channel.
   - Tabular: ensure z-scored inputs; handle missing values if any.
3) **Baselines**:
   - RGB-only: pretrained ResNet/ConvNeXt/ViT with linear classifier; cross-entropy, class weights if needed.
   - Tabular-only: MLP or gradient boosting (e.g., XGBoost/LightGBM if allowed).
4) **Fusion**:
   - Early fusion: concatenate CNN embedding with tabular features, then MLP head.
   - Late fusion: weighted average or learned gating over separate RGB/tabular logits.
   - Intermediate fusion: project tabular features and cross-attend with image tokens (ViT-based).
5) **Training details**:
   - Optimizer: AdamW; cosine or step LR; weight decay.
   - Regularization: dropout, label smoothing, Mixup/CutMix (if helpful), class weighting or focal loss for imbalance.
   - Batch sampling: respect geographic split; avoid leakage.
6) **Evaluation**:
   - Metrics: overall accuracy, macro/micro F1, per-class F1, confusion matrix, expected calibration error (ECE), reliability diagram.
   - Ablations: per-theme SWECO subsets; modality comparison (RGB vs tabular vs fusion).
7) **Inference & Mapping**:
   - Apply best model on held-out test and on any additional tiles; if producing maps, export GeoTIFF/GeoPackage with predicted class per pixel/location.

## Reproducibility Checklist
- Fix seeds; log versions and hyperparameters.
- Save best checkpoints and evaluation artifacts (confusion matrices, reliability plots).
- Document dataset paths and any preprocessing caches.

## References
- ProjectDescription.pdf (ENV-540, Fall 2025, topic 5).
- swissIMAGE orthophotos: https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10
- SWECO25 database: Kuelling et al., 2024. SWECO25: a cross-thematic raster database for ecological research in Switzerland.
- EUNIS Habitat Classification: Chytry et al., 2020.
