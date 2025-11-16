[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/full-resolution-network-and-dual-threshold/retinal-vessel-segmentation-on-drive)](https://paperswithcode.com/sota/retinal-vessel-segmentation-on-drive?p=full-resolution-network-and-dual-threshold)[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/full-resolution-network-and-dual-threshold/retinal-vessel-segmentation-on-chase_db1)](https://paperswithcode.com/sota/retinal-vessel-segmentation-on-chase_db1?p=full-resolution-network-and-dual-threshold)
# FR-UNet
This repository is the official PyTorch code for the paper ['Full-Resolution Network 
and Dual-Threshold Iteration for Retinal Vessel and Coronary Angiograph Segmentation'](https://ieeexplore.ieee.org/abstract/document/9815506).


<div align="center">
  <img src="figs/FR-UNet.png" width="100%">
</div>

 
## Prerequisites
 

 
Download our repo:
```
git clone https://github.com/lseventeen/RF-UNet.git
cd RF-UNet
```
Install packages from requirements.txt
```
pip install -r requirements.txt
```
 
## Datasets processing
Choose a path to create a folder with the dataset name and download datasets [DRIVE](https://www.dropbox.com/sh/z4hbbzqai0ilqht/AAARqnQhjq3wQcSVFNR__6xNa?dl=0),[CHASEDB1](https://blogs.kingston.ac.uk/retinal/chasedb1/),[STARE](https://cecas.clemson.edu/~ahoover/stare/probing/index.html),[CHUAC](https://figshare.com/s/4d24cf3d14bc901a94bf), and [DCA1](http://personal.cimat.mx:8181/~ivan.cruz/DB_Angiograms.html). Type this in terminal to run the data_process.py file
 
```
python data_process.py -dp DATASET_PATH -dn DATASET_NAME
```

### Consolidated `ALL` dataset

If you already merged multiple datasets into a single folder named `ALL`, organize the data as:

```
ALL/
  Original/      # 1.png, 2.png, ...
  Segmented/     # 1_segmented.png, ...
  Mask/          # 1_mask.png, ... (optional per sample)
```

Run preprocessing, apply CLAHE + FOV masking, and create the train/test pickles with:

```
python data_process.py -dp PATH_TO/ALL -dn ALL -tr 0.85
```

`-tr` controls the train/test split ratio (train share). Missing masks are skipped automatically and augmented samples always use the same random flip/rotation for both the image and its segmentation.

To patchify the remaining 15% hold-out tiles (used by the 100% training regime) once the primary preprocessing is finished, run:

```
python data_process.py -dp PATH_TO/ALL -dn ALL -tr 0.85 -m holdout
```

Holdout patch pickles restart their filenames at `img_patch_0.pkl` but live under `holdout_pro/`, so despite the reset they only contain the samples that follow the first 85 percent (e.g., everything after training index 745,735 resides exclusively in `holdout_pro`).
 
## Training
Type this in terminal to run the train.py file
 
```
python train.py -dp DATASET_PATH
```

### Automated ALL dataset pipelines

The repository now ships with two automation scripts that train, evaluate, and export artifacts in a single command:

```
# Train on the base 85% patch split and evaluate on test_pro
python train_85.py -dp PATH_TO/ALL --device cuda

# Train on the full 100% patch pool (85% base + 15% holdout) and evaluate on test_pro
python train_100.py -dp PATH_TO/ALL --device cuda
```

Pass `--device cuda` whenever you have a GPU available (e.g., your rented NVIDIA A100). When CUDA is absent, the scripts automatically fall back to CPU so you can dry-run locally before moving to the cloud.

Each run creates a timestamped subfolder under `runs/train_85` or `runs/train_100` containing:

- the copied `config.yaml` used for the run
- the final checkpoint (`checkpoint-epochXX.pth`)
- `test_metrics.json` plus raw/normalized confusion-matrix PNGs
- `run_summary.json` with paths to every artifact

Use `--output-root` to change where these artifacts are written and `--val` to reserve 10% of the base training patches for validation.

### Offline preprocessing → cloud training workflow

Run preprocessing on your local workstation (example shown for Windows PowerShell, adjust interpreter path as needed):

```powershell
cd d:/FR-UNet/FR-UNet
C:/Users/RC/AppData/Local/Programs/Python/Python312/python.exe data_process.py -dp d:/FR-UNet/FR-UNet/ALL -dn ALL -tr 0.85 -m training test
C:/Users/RC/AppData/Local/Programs/Python/Python312/python.exe data_process.py -dp d:/FR-UNet/FR-UNet/ALL -dn ALL -tr 0.85 -m holdout
```

This generates `training_pro/`, `test_pro/`, and `holdout_pro/` inside the `ALL` folder. Compress (zip/tar) that `ALL` directory and upload it to your cloud VM.

Once unpacked on the cloud machine, install requirements and run the desired pipeline:

```bash
cd /path/to/FR-UNet
pip install -r requirements.txt

# Base 85 percent training regime
python train_85.py -dp /path/to/ALL

# Full 100 percent training regime (requires holdout_pro)
python train_100.py -dp /path/to/ALL
```

Each command leaves a timestamped folder under `runs/train_85` or `runs/train_100` containing the full checkpoint directory, copied `config.yaml`, evaluation metrics, and confusion-matrix images so every weight file is saved per run.
## Test
Type this in terminal to run the test.py file
 
```
python test.py -dp DATASET_PATH -wp WEIGHT_FILE_PATH
```
We have prepared the pre-trained models for both datasets in the folder 'pretrained_weights'. To replicate the results in the paper, directly run the following commands
```
python test.py -dp DATASET_PATH -wp pretrained_weights/DATASET_NAME
```


 
## License
 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
 
