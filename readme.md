# Domain Adaptation via Iterative Reconstruction

A compact repository for test-time domain adaptation using iterative reconstruction for the following [paper](https://openreview.net/forum?id=9wIPViPG9F&referrer=%5Bthe%20profile%20of%20Hrvoje%20Bogunovi%C4%87%5D(%2Fprofile%3Fid%3D~Hrvoje_Bogunovi%C4%871)) accepted at Medical Imaging with Deep Learning 2026 (MIDL). The method adapts a pretrained model at test time using two modulators (EntropyModulator and NormModulator) and evaluates performance on downstream segmentation tasks.

## Highlights
- Test-time adaptation driven by iterative reconstruction.
- Core components:
  - EntropyModulator — adapts confidence/entropy during inference.
  - NormModulator — adjusts normalization statistics for domain shift.
  - segmodel — a generic downstream segmentation/retouch network.
- Based on the GARD repository (Fazekas et al.): https://github.com/ABotond/GARD

## Data preprocessing
- RETOUCH dataset images were resized to 512×512 using: `skimage.transform.resize` with default parameters (default parameters were used).
- Store the images in a folder split into `Input` for the images and `Label` for the segmentation mask as evaluation. I named the files Pat_{pat_idx}_{slice_idx}.png. 

## Running experiments
1. Train a GARD model using the repository of Fazekas.
2. Update dataset and model folder paths in the configuration or scripts.
3. Run the test-time adaptation script:

python test_gamma_for_adaption.py

## Repository layout (important files)
- EntropyModulator — adaptation logic for entropy-based updates.
- NormModulator — normalization/statistics adaptation.
- segmodel — downstream model definition and evaluation code.
- test_gamma_for_adaption.py — entry point for adaptation experiments.

## Notes
- Change folder names and paths to match your local dataset layout before running experiments.
- This project focuses on the adaptation modules; refer to the baseline GARD repository for original training/evaluation code and inspiration.

## Citation / Reference

I heavily relied on the following repository by Fazekas et al.: https://github.com/ABotond/GARD


If you use this repository, please cite the following paper:

```
@inproceedings{pinetz2026exploiting,
  title={Exploiting Intermediate Reconstructions in Optical Coherence Tomography for Test-Time Adaption of Medical Image Segmentation},
  author={Pinetz, Thomas and Hucke, Veit and Bogunovi{\'c}, Hrvoje},
  booktitle={Medical Imaging with Deep Learning},
  year={2026}
}
```
