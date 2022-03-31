# neuralg
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about">About</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#goals">Goals</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
     <ul>
        <li><a href="#support">Support</a></li>
        <li><a href="#small-example">Small example</a></li>
        <li><a href="#training-distribution">Training distribution</a></li>
      </ul>
    </li>
  <ol>
</details>

<!-- ABOUT THE PROJECT -->
## About 

The neuralg module is a neural network based collection of approximators to common numerical linear algebra operations. It allows utilizing GPUs for differentiable and efficient computation used together with [PyTorch](https://pytorch.org/). 

The software is free to use and is designed for the machine learning community in general, and users focusing on topics involving numerical linear algebra in particular.


### Built With

This project is built with the following packages:

* [PyTorch](https://pytorch.org/), implying full differentiability and can be used for machine learning
<!--* [conda](https://docs.conda.io/en/latest/), which will take care of all requirements for you.-->

<!-- GOALS -->
## Goals

* **Proof of concept**: Recent years of rapid advances in machine learning aside, neural network models are yet to reach competitive results in the field of numerical linear algebra. Some attention has been paid to the subject, e.g. with parameter rich transformers as in [insertreference](https://arxiv.org/pdf/2112.01898.pdf). neuralg serves as a demonstration of a competitive small-scale approach, with the goal of mitigating issues with memory and time complexity related to larger models.
* **Supporting science**: Linear algebra problems serve as fundamental computational components in countless science and engineering applications. Eigenvalue and singular value decompositions, solving linear system of equations and matrix inversion appear as essential parts of solutions in  optimization, dynamical systems, signal processing etc. Ultimately, neuralg aims to provide useful tools to researchers within these fields, with a focus on parallell computation.
* **Addressing efficient vectorization**: The nature of classical numerical algorithms for linear algebra operations are often iterative and difficult to vectorize efficiently on GPUs and specialized machine learning hardware. Existing built-in libraries often synchronize with CPU, which can severly slow down computation. To this end, neuralg aims to allow users to exploit the computational benefits from GPU parallelization on targeted hardware.
<!-- GETTING STARTED -->
## Getting Started

What follows a short guide for setting up neuralg on your device.

### Prerequisites

<!-- Should we have conda or not? -->
### Installation
<!-- 
To install neuralg via conda run

   ```sh
   conda install torchquad -c conda-forge -c pytorch
   ```
-->

To install neuralg via pip, run
   ```sh
   pip install neuralg
   ```

NB Note that *pip* will **not** set up PyTorch with CUDA and GPU support. <!--Recommend to use *conda*? -->

<!--
**GPU Utilization**

With *conda* you can install the GPU version of PyTorch with `conda install pytorch cudatoolkit -c pytorch`. 
For alternative installation procedures please refer to the [PyTorch Documentation](https://pytorch.org/get-started/locally/). -->

<!-- USAGE EXAMPLES -->
## Usage
### Support
Built with PyTorch and targeting GPU utilization, neuralg only supports input of type `torch.Tensor`. The current version of neuralg supports real valued input matrices of float dtype. Supported outputs are float and cfloat dtypes.
### Small example
The neuralg module is designed to resemble existing, commonly used numerical linear algebra libraries. What follows here is a small example showing how neuralg can be used to find the eigenvalues of a batch of random matrices. For a more elaborate and interactive example, please refer to the jupyter notebook [insertlinktoexampleproblemnotebook] 

```python

import torch 
import neuralg 
from neuralg import eig 
# Enable GPU support if available 
neuralg.set_up_torch(enable_cuda = True)

# Sample a batch of matrices with uniform iid coefficients
# Note that neuralg only supports input of tensor type 
batch_size, matrix_size = 10000, 15
matrix_batch = torch.rand(batch_size,matrix_size,matrix_size)

# Call neuralg to approximate eigenvalues 
eigenvalues = eig(matrix_batch)

```
<!-- All available linear algebra operations are [insertlink].-->
### Training distribution
Current available models have been trained and evaluated on random matrices. neuralg also supports training models from scratch or re-training and fine tuning existing models, depending on specific user applications. Please refer to [insertlinktotrainingtutorialnotebook] for a thorough how-to guide.
#### eig
<!-- Subject to change-->
 <span style="color:red"> The underlying training distribution for models supporting symmetric matrices are symmetric matrices with i.i.d. normally distributed eigenvalues $ \lambda_i \sim N(0, \sigma_{train}^2)$ and eigenvectors uniformly distributed on the unit sphere. Models supporting non-symmetric matrices were trained on real valued matrices $X$ with i.i.d. uniformly distributed elements $ X_{ij} \sim U(-x,x)$. 
  </span>
