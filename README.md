# PreSTA

This is the official PyTorch implementation for the CVPR 2026 paper
**"Rethinking Knowledge Transfer in Image Quality Assessment: A Perceptual Preference Structure Alignment Perspective"**.


## Requirement

+ Python 3.8.20
+ PyTorch 2.4.1
+ torchvision 0.19.1
+ timm 1.0.11


## Usage

### 1. Extract feature statistics

Run `extract_features.py` to extract hierarchical ImageNet features for both
the source dataset and target dataset. The same backbone should be used in all
subsequent steps. Using "KADID-10k → LIVEC" as an example.

```bash
python extract_features.py \
  --root /path/to/root \
  --dataset kadid-10k livec \
  --backbone swin_base_patch4_window7_224 \
  --device 0
```

The extracted feature files will be saved to `fs/`.

Supported backbones:

+ `swin_base_patch4_window7_224`
+ `resnet50`


### 2. Run PreSTA source sample selection

Use `PreSTA.py` to select source samples by aligning perceptual preference
structures between the source and target domains.

```bash
python PreSTA.py \
  --source-db kadid-10k \
  --target-db livec \
  --feature-dir fs \
  --backbone swin_base_patch4_window7_224 \
  --output-dir SpecSelBatch
```

### 3. Train and test

Use `train_test_IQA_cross.py` to run cross-database BIQA experiments.

```bash
python train_test_IQA_cross.py \
  --root /path/to/root \
  --dataset kadid-10k \
  --test_db livec \
  --backbone swin_base_patch4_window7_224 \
  --device 0
```

## Citation

If you find our code helpful for your research, please consider citing our
paper. The BibTeX entry will be updated after the proceedings version is
available.

```bibtex
@inproceedings{presta2026rethinking,
  title={Rethinking Knowledge Transfer in Image Quality Assessment: A Perceptual Preference Structure Alignment Perspective},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
