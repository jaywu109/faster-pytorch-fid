# Faster FID score for PyTorch

This is a modified implementation of [pytorch-fid](https://github.com/mseitzer/pytorch-fid). The main purpose of this refined version is to replace the computation of [matrix square root](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html) with PyTorch in GPU that is originally implemented with SciPy in CPU. It can accelerate the calculation of Fréchet Inception Distance from 5 minutes to 30 seconds with the help of GPU in my environment (when `dims=2048`).

## Installation

Install from [pip](https://pypi.org/project/pytorch-fid/):

```
pip install pytorch-fid
```

Requirements:
- python3
- pytorch
- torchvision
- pillow
- numpy
- scipy

## Usage

Please refer to original pytorch-fid repo for detailed usage.

## Detailed Refinement

In some circumestances, The orginal pytorch-fid takes a very long time to computer the FID metrics. After diving into the implementation, I noticed the major bottelneck is for calculating matrix square root with SciPy (scipy.linalg.sqrtm) as below:

https://github.com/jaywu109/faster-pytorch-fid/blob/aff60dcff18a927f640042bf08021f643828033f/fid_score.py#L188

A detailed investigation showd that there exist multiple numpy dot product operations in the [sqrtm](https://github.com/scipy/scipy/blob/v1.10.1/scipy/linalg/_matfuncs_sqrtm.py#L117-L210). When using default dimensionality of features of 2048 (final average pooling features), it would be really slow to compute dot product for metrics with size of `2048x2048` with CPU. 

My firt intuition is to replace the sqrtm in SciPy with the one from CuPy with GPU support. However, it's currently not supported for now according to the [comparison table](https://docs.cupy.dev/en/stable/reference/comparison.html). To improve the computation efficiency with minimal effort for refactoring the code, I decided to merly change the dot product operation with `torch.matmul` and put the operation on GPU in sqrtm function that cost most of the time, instead of changing all the numpy operaitons in sqrtm with torch equvilences that may not result in significat improvement.


## Citing

If you use this repository in your research, consider citing it using the following Bibtex entry:

```
@misc{Seitzer2020FID,
  author={Maximilian Seitzer},
  title={{pytorch-fid: FID Score for PyTorch}},
  month={August},
  year={2020},
  note={Version 0.3.0},
  howpublished={\url{https://github.com/mseitzer/pytorch-fid}},
}
```

## License

This implementation is licensed under the Apache License 2.0.

FID was introduced by Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler and Sepp Hochreiter in "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium", see [https://arxiv.org/abs/1706.08500](https://arxiv.org/abs/1706.08500)

The original implementation is by the Institute of Bioinformatics, JKU Linz, licensed under the Apache License 2.0.
See [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR).
