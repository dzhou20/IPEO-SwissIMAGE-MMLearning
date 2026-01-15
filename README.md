# IPEO-SwissIMAGE-MMLearning

This project develops a multimodal deep learning framework for Swiss ecosystem mapping, combining swissIMAGE RGB aerial imagery and SWECO25 environmental variables to predict 17 EUNIS ecosystem classes. The study compares image-only, tabular-only, and multimodal fusion models, and analyzes the contribution of different environmental variable groups under a geographically independent evaluation setting.

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

Switch backbone:
```
python train.py --mode image --backbone convnext_tiny
python train.py --mode image --backbone efficientnet_b0
```

Notes:
- `--group` is required in fusion mode. Valid groups are the keys in `sweco_group_of_variables.py`.
- `--group all` merges all SWECO groups and removes duplicate variable names.
- Add `--pretrained` to load pretrained weights (requires local cache or network access).

## Ablation Study

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
| D1 | ConvNeXt-Tiny   | Yes            | All             | None                                            | Early  |
| D2 | EfficientNet-B0 | Yes            | All             | None                                            | Early  |
| E1 | ResNet18        | Yes            | All             | Pretrained = False                              | Early  |
| G1 | ResNet18        | Yes            | All             | lr = 1e-5                                       | Early  |
| G2 | ResNet18        | Yes            | All             | lr = 1e-3                                       | Early  |
| G3 | ResNet18        | Yes            | All             | lr = 5e-4                                       | Early  |
| G4 | ResNet18        | Yes            | All             | weight_decay = 1e-2                             | Early  |
| G5 | ResNet18        | Yes            | All             | lr = 5e-4                                       | Gated  |

## References
- ProjectDescription.pdf (ENV-540, Fall 2025, topic 5).
- swissIMAGE orthophotos: https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10
- SWECO25 database: Kuelling et al., 2024. SWECO25: a cross-thematic raster database for ecological research in Switzerland.
- EUNIS Habitat Classification: Chytry et al., 2020.