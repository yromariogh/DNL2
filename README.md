# [Deep Learned Non-Linear Propagation Model Regularizer for Compressive Spectral Imaging](https://ieeexplore.ieee.org/abstract/document/10587094)

## Abstract

Coded aperture snapshot spectral imager (CASSI), efficiently captures 3D spectral images by sensing 2D projections of the scene. While CASSI offers a substantial reduction in acquisition time, compared to traditional scanning optical systems, it requires a reconstruction post-processing step. Furthermore, to obtain high-quality reconstructions, an accurate propagation model is required. Notably, CASSI exhibits a variant spatio-spectral sensor response, making it difficult to acquire an accurate propagation model. To address these inherent limitations, this work proposes to learn a deep non-linear fully differentiable propagation model that can be used as a regularizer within an optimization-based reconstruction algorithm. The proposed approach trains the non-linear spatially-variant propagation model using paired compressed measurements and spectral images, by employing side information only in the calibration step. From the deep propagation model incorporation into a plug-and-play alternating direction method of multipliers framework, our proposed method outperforms traditional CASSI linear-based models. Extensive simulations and a testbed implementation validate the efficacy of the proposed methodology.

## Test
```
python real0.py --sX=1 --l1=1.0 --map=15 --sCA=CA_ideal --PM=Simple --denoiser=UNet --all=False --lr=0.01 --init=Transpose --net=C4 --interrupt=True --gpu="0"
```

## Train
```
python train.py --net=C4 --retrain=False --evaluate=false --epochs=1000 --batch=1 --lr=1e-3
```

## Requirements
- Python == 3.9
- Tensorflow == 2.7

The exact conda environment can be installed using
```
conda env create -f tf27_gpu_no_build.yml
```

## Citation
```
@article{gualdron2024deep,
  title={Deep Learned Non-Linear Propagation Model Regularizer for Compressive Spectral Imaging},
  author={Gualdr{\'o}n-Hurtado, Romario and Arguello, Henry and Bacca, Jorge},
  journal={IEEE Transactions on Computational Imaging},
  year={2024},
  publisher={IEEE}
}
```
