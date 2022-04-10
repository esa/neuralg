# neuralg
<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/gomezz/neuralg">
    <img src="test_neuralg_logo.png" alt="Logo"  width="240" height="120">
  </a>
  <p align="center">
    Neural network approximators of linear algebra operations on GPU with PyTorch
    <br />
    <a href="https://github.com/gomezzz/neuralg/notebooks/example_problem.ipynb">View Example notebook</a>
  </p>
</p>


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
        <li><a href="#test">Test</a></li>
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

<!-- GOALS -->
## Goals

* **Proof of concept**: Recent years of rapid advances in machine learning aside, neural network models are yet to reach competitive results in the field of numerical linear algebra. Some attention has been paid to the subject, e.g. with parameter rich transformers as in [here](https://arxiv.org/pdf/2112.01898.pdf). neuralg serves as a demonstration of a competitive small-scale approach, with the goal of mitigating issues with memory and time complexity related to larger models.
* **Supporting science**: Linear algebra problems serve as fundamental computational components in countless science and engineering applications. Eigenvalue and singular value decompositions, solving linear system of equations and matrix inversion appear as essential parts of solutions in  optimization, dynamical systems, signal processing etc. Ultimately, neuralg aims to provide useful tools to researchers within these fields, with a focus on parallell computation.
* **Addressing efficient vectorization**: The nature of classical numerical algorithms for linear algebra operations are often iterative and difficult to vectorize efficiently on GPUs and specialized machine learning hardware. Existing built-in libraries often synchronize with CPU, which can severly slow down computation. To this end, neuralg aims to allow users to exploit the computational benefits from GPU parallelization on targeted hardware.
<!-- GETTING STARTED -->
## Getting Started

What follows a short guide for setting up neuralg on your device.

### Prerequisites
<!-- Should there be anything in this entry? Like Install CUDA, if your machine has a CUDA-enabled GPU?  -->
### Installation

To install neuralg via pip, run
   ```sh
   pip install neuralg
   ```

NB Note that *pip* will **not** set up PyTorch with CUDA and GPU support. <!-- Is this true? -->

**GPU Utilization**
To set up the GPU version of PyTorch, please refer to installation procedures at [PyTorch Documentation](https://pytorch.org/get-started/locally/)

### Test 
After cloning the repository, developers can check the functionality of `neuralg` by running the following command in the root directory: <!-- At least I think so? Or should it be in the tests directory?!-->

```sh
pytest
```

<!-- USAGE EXAMPLES -->
## Usage
### Support
Built with PyTorch and targeting GPU utilization, neuralg only supports input of type `torch.Tensor`. The current version of neuralg supports real valued input matrices of float dtype. Supported outputs are float and complexfloat dtypes.
### Small example
The neuralg module is designed to resemble existing, commonly used numerical linear algebra libraries. Below is a small example showing how neuralg can be used to approximate the eigenvalues of a batch of random matrices. For a more elaborate and interactive example, please refer to the jupyter notebook [example problem](https://github.com/gomezzz/neuralg/notebooks/example_problem.ipynb). <!-- Change this link when public repo-->

```python

import torch 
import neuralg 
from neuralg import eig 
# Enable GPU support if available 
neuralg.set_up_torch(torch_enable_cuda = True)

# Sample a batch of matrices with uniform iid coefficients
# Note that neuralg only supports input of tensor type 
batch_size, matrix_size = 10000, 5
matrix_batch = torch.rand(batch_size,matrix_size,matrix_size)

# Call neuralg to approximate eigenvalues 
eigvals = eig(matrix_batch) # outputs are torch.complexfloat dtype

# Construct symmetric matrices
symmetric_matrix_batch = torch.triu(matrix_batch, 0) + torch.transpose(torch.triu(matrix_batch, 1),1,2) 
# Symmetric matrices have specialized appproximators
eigvals_sym =  eig(symmetric_matrix_batch, symmetric = True) # output are torch.float dtype

```
<!-- All available linear algebra operations are [insertlink].-->
### Training distributions
Current available models have been trained and evaluated on random quadratic matrices. For details, see specifications for corresponding operations and matrix types. Additionally, neuralg supports training models from scratch or re-training and fine tuning existing models, depending on specific user applications. Please refer to the [tutorial](https://github.com/gomezzz/neuralg/notebooks/training_tutorial.ipynb) for a thorough how-to guide. <!-- Change this link when public repo-->
|     **function call**     | **supported sizes** |                                                              **training matrix distribution**                                                              |
|:-------------------------:|:-------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------:|
| `eig(x,symmetric = True)` |        [3,20]       | Real valued symmetric with i.i.d. centered normally distributed eigenvalues with variance 100/3 and eigenvectors uniformly distributed on the unit sphere. |
|    `eig(x,real = True)`   |        [3,10]       |                         Real valued asymmetric matrices with i.i.d. centered normally distributed eigenvalues with variance 100/3.                         |
|          `eig(x)`         |        [3,5]        |                                        Real valued matrices with i.i.d. uniformly distributed elements on [-10,10].                                        |
|          `svd(x)`         |        [3,20]       |                                        Real valued matrices with i.i.d. uniformly distributed elements on [-10,10].                                        |

    
## License

Distributed under the GPL-3.0 License. See [LICENSE](https://github.com/gomezzz/neuralg/LICENSE) for more information.
