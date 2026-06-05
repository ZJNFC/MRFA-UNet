# MRFA-UNet for 3-D Seismic Fault Segmentation

This repository contains the source code used for training and applying
MRFA-UNet, a task-specific 3-D encoder-decoder network for seismic fault
segmentation. The code also includes several baseline models used for comparison
in the manuscript.

## Repository Structure

```text
.
|-- apply.py                  # Field-data application and visualization
|-- train.py                  # Model training and validation
|-- data/
|   |-- data_augmentation.py  # 3-D data augmentation
|   `-- data_loader.py        # Seismic fault dataset loader
|-- loss/
|   `-- FocalTverskyLoss.py   # Focal Tversky loss
`-- model/
    |-- MRFA_UNet.py          # Proposed MRFA-UNet
    |-- segnet.py             # 3-D SegNet baseline
    |-- transunet.py          # TransUNet baseline
    `-- unet.py               # 3-D U-Net baseline
```

## Requirements

The code is implemented in Python with PyTorch. The main dependencies are:

```text
python
numpy
matplotlib
tqdm
torch
torchmetrics
tensorboard
```

A CUDA-enabled GPU is recommended for training 3-D seismic volumes.

## Data

The datasets used in the manuscript are public datasets and are not redistributed
in this repository. Please download the datasets from their original public
sources and arrange the training data in the following format:

```text
data/
|-- train/
|   |-- seis/
|   |   |-- sample_001.dat
|   |   `-- ...
|   `-- fault/
|       |-- sample_001.dat
|       `-- ...
`-- validation/
    |-- seis/
    |   |-- sample_201.dat
    |   `-- ...
    `-- fault/
        |-- sample_201.dat
        `-- ...
```

The default data loader reads binary `.dat` files as `float32` arrays with a
default volume size of `(128, 128, 128)`. The seismic volume and the corresponding
fault label should have the same file name in the `seis` and `fault` folders.

## Training

Before training, set the dataset path in `train.py`:

```python
DATA_DIR = r"data"
```

Then run:

```bash
python train.py
```

The default training configuration uses MRFA-UNet. Other available model types
are:

```text
mrfa_unet
unet
segnet
transunet
```

Training checkpoints and TensorBoard logs are saved under:

```text
result/<model_type>/<date>/<time>/
```

## Field-Data Application

The script `apply.py` is used for applying a trained model to field seismic data
and saving slice-based visualizations.

Before running the script, set the trained model path:

```python
model_weight = r"path/to/model.pth"
```

For F3 field-data application, set the data directories in `F3_DATASET`:

```python
F3_DATASET = {
    "name": "F3",
    "seis_dir": r"path/to/f3/seis",
    "fault_dir": r"path/to/f3/fault",
    "volume_shape": (512, 384, 128),
    ...
}
```

The visualization axes and slice indices can be adjusted near the bottom of
`apply.py`:

```python
AXES = ["z"]
INDICES = {"z": [110]}
```

Then run:

```bash
python apply.py
```

Output figures are saved under:

```text
result/<model_type>/output/<date>/<time>/
```

## Notes

- Public seismic datasets are not included in this repository.
- The scripts contain default paths and parameters that should be adjusted
  according to the local dataset location and experimental setting.
- The code is intended for research use in 3-D seismic fault segmentation.

## Citation

If you use this code in your research, please cite the corresponding manuscript.
