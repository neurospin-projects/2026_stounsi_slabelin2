# Geometric-to-Semantic Spherical Transfer Learning for Cortical Sulci Labeling

Official implementation for exhaustive cortical sulci labeling in data-scarce regimes (N=62). The framework utilizes a self-supervised spherical encoder pre-trained on ≈ 30,000 UK Biobank subjects, adapted for semantic fine-tuning via a soft-initialized Topological Prior Injector (TPI).

## Core Features
* **Hybrid Spherical U-Net**: Integrates pre-trained geometric features with explicit semantic lines (fundi).
* **Topological Prior Injector (TPI)**: A 1x1 convolution that bridges the 2-channel pre-training (C, S) and 3-channel downstream input (C, S, L) without catastrophic forgetting.
* **Flexible Input Configuration**: Input channels are not fixed and can be customized in `config.yaml`. You can stack as many surface descriptors as needed. The first channel (index 0) is explicitly reserved for sulcal lines (fundi). You can dynamically include or exclude this semantic channel during training.
* **Surface-Native Pipeline**: Custom KDTree-based 3D augmentations and graph-based morphological post-processing.

## Repository Structure
* `config.yaml`: Centralized Hydra configuration managing hyperparameters and flexible inputs.
* `main.py`: Entry point handling K-Fold cross-validation and differential learning rates.
* `models/model.py`: Core architectures, including `SphericalUNetFromPretrained` and model baselines. For Dinov3 please replace by your token.
* `training/`: Training loop and geometric augmentations.
* `evaluation/eval.py`: Specialized metrics (Dice, ESI, Elocal) and networkX-based topological graph cleaning.

## Installation

```bash
git clone ***
cd ***

pip install -r requirements.txt
```

## Usage

Define your custom inputs and parameters in `config.yaml`:

```yaml
data:
  use_lines: True 
  in_channels: 3
  inputs:
    - "{hemi}.topological_lines.curv"
    - "{hemi}.ico{order}.curv"
    - "{hemi}.ico{order}.sulc"
```

**Run a standard transfer learning training:**

```bash
python main.py experiment_name="transfer_run" model.name="PretrainedUNet"
```
Please note that SphericalUnet , Dinov2 , Dinov3 , 2D-Unet are also available.
 
**Override hyperparameters on the fly:**

```bash
python main.py data.use_lines=False training.encoder_lr=1e-5 cross_validation.k_folds=4
```
