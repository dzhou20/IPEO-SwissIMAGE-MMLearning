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
- Use the provided Conda environment file: `environment.yml`.

Example setup:
```
conda env create -f environment.yml
conda activate ipeo-env
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

## Run The Baselines
All training runs automatically evaluate on the test set and save the confusion matrix (CSV + PNG) under `outputs/<run_name>/`.

Image-only baseline (ResNet18, 200x200 input):
```
python train.py --mode image --backbone resnet18
```

Tabular-only baseline (all SWECO variables):
```
python train.py --mode tabular --group all
```

Early fusion with a single SWECO group (example: bioclim):
```
python train.py --mode fusion --group bioclim --backbone resnet18
```

Early fusion with all SWECO variables (merged and de-duplicated):
```
python train.py --mode fusion --group all --backbone resnet18
```

Early fusion with a SWECO group combination (example: bioclim + vege):
```
python train.py --mode fusion --group "bioclim, vege" --backbone resnet18
```

Switch backbone to ViT:
```
python train.py --mode image --backbone vit
```

Notes:
- `--group` is required in fusion mode. Valid groups are the keys in `sweco_group_of_variables.py`.
- `--group all` merges all SWECO groups and removes duplicate variable names.
- Add `--pretrained` to load pretrained weights (requires local cache or network access).

## Reproducibility Checklist
- Fix seeds; log versions and hyperparameters.
- Save best checkpoints and evaluation artifacts (confusion matrices, reliability plots).
- Document dataset paths and any preprocessing caches.

## References
- ProjectDescription.pdf (ENV-540, Fall 2025, topic 5).
- swissIMAGE orthophotos: https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10
- SWECO25 database: Kuelling et al., 2024. SWECO25: a cross-thematic raster database for ecological research in Switzerland.
- EUNIS Habitat Classification: Chytry et al., 2020.


## Ablation Study

ResNet18T denotes a ResNet18 backbone initialized with ImageNet pretrained weights.

**Default Training Configuration**

Unless otherwise specified, all experiments are conducted under the following default settings:

- Maximum epochs: 100
- Minimum epochs: 30
- Batch size: 32
- Learning rate: 1e-4
- Weight decay: 1e-4
- Scheduler: ReduceLROnPlateau (patience = 5), monitored on validation macro-F1 score
- Loss function: Cross-Entropy Loss with class weights
- Early stopping: triggered after 10 consecutive epochs without improvement in validation loss
- Fusion mode: Early fusion
- Pretrained: True

| ID | Image Backbone  | Image Features | SWECO Features  | Strategy                                        | Fusion |
| -- | --------------- | -------------- | --------------- | ----------------------------------------------- | ------ |
| A0 | ResNet18        | Yes            | None            | None                                            | No     |
| A1 | ResNet18        | Yes            | All             | None                                            | Early  |
| A2 | ResNet18        | Yes            | geol            | None                                            | Early  |
| A3 | ResNet18        | Yes            | edaph           | None                                            | Early  |
| A4 | ResNet18        | Yes            | vege            | None                                            | Early  |
| A5 | ResNet18        | Yes            | bioclim         | None                                            | Early  |
| A6 | ResNet18        | Yes            | lulc_grasslands | None                                            | Early  |
| A7 | ResNet18        | Yes            | lulc_all        | None                                            | Early  |
| A8 | ResNet18        | Yes            | hydro           | None                                            | Early  |
| A9 | ResNet18        | Yes            | population      | None                                            | Early  |
| B1 | ResNet18        | No             | All             | None                                            | No     |
| C1 | ResNet18        | Yes            | All             | None                                            | Gated  |
| C2 | ResNet18        | Yes            | All             | None                                            | Late   |
| D1 | ViT             | Yes            | All             | None                                            | Early  |
| D2 | ConvNeXt-Tiny   | Yes            | All             | None                                            | Early  |
| D3 | EfficientNet-B0 | Yes            | All             | None                                            | Early  |
| E1 | ResNet18        | Yes            | All             | Pretrained = False                              | Early  |
| F1 | ResNet18        | Yes            | All             | Stage-wise Fine-tuning (Progressive Unfreezing) | Early  |
| G1 | ResNet18        | Yes            | All             | lr = 1e-5                                       | Early  |
| G2 | ResNet18        | Yes            | All             | lr = 1e-3                                       | Early  |
| H1 | ResNet18        | Yes            | All             | Two-stage classification model                  | Early  |

