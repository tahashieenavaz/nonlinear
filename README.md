# Non-Linear

A hand-curated collections of activations functions for deep learning research.

## Installation

```bash
pip install nonlinear
```

## Important Note

In order to gain the best performance speed **always** use `PyTorch` with a version greater than 2.0.0 and use `torch.compile` on the whole model that uses activations. This will utilize under the hood triton kernels to speed up the training and inference speeds.

# Citation

```bibtex
@misc{2402.09092,
  Title = {Three Decades of Activations: A Comprehensive Survey of 400 Activation Functions for Neural Networks},
  Author = {Vladimír Kunc and Jiří Kléma},
  Year = {2024},
  Eprint = {arXiv:2402.09092},
}
```

```bibtex
@misc{1505.00853,
  Title = {Empirical Evaluation of Rectified Activations in Convolutional Network},
  Author = {Bing Xu and Naiyan Wang and Tianqi Chen and Mu Li},
  Year = {2015},
  Eprint = {arXiv:1505.00853},
}
```