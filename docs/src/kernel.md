# Kernels (Covariance functions)

The **kernel function** or **covariance function** is a crucial part of Gaussian Processes. It determines the covariance matrices between set of points, and its behaviour and parameters determines almost completely a GP behaviour.

## Kernels available
To get a short introduction of some covariance functions available one can look at these
[Slides from Rasmussen](http://mlg.eng.cam.ac.uk/teaching/4f13/1819/covariance%20functions.pdf)

We define $\theta_i$ as the lengthscale (in case of `IsoKernel` $\theta_i=\theta\;\forall i$) and $\sigma$ is the variance
In this package covariance functions are progressively added for now the available kernels are :

- RBF Kernel or Squared Exponential Kernel


$$k(x,x') = \sigma \exp\left(-\frac{1}{2}\sum_{i=1}^D\frac{(x_i-x_i')^2}{\theta_i^2}\right)$$

- Matern Kernel

$$k(x,x') = \sigma\frac{2^{1-\nu}}{\Gamma(\nu)}\(\sqrt{2\nu}\frac{d}{\rho}\right)^\nu K_\nu\left(\sqrt{2\nu}\frac{d}{\rho}\right)$$

More are coming, check the [github projects](https://github.com/theogf/AugmentedGaussianProcesses.jl/projects/1) for updates .

However the module for kernels should be replaced soon by [MLKernels.jl](https://github.com/trthatcher/MLKernels.jl)
## Hyperparameter optimization

The advantage of Gaussian Processes is that it is possible to optimize all the hyperparameters of the model by optimizing the lower bound on the loglikelihood. One can compute the gradient of it and apply a classical gradient descent algorithm.

Unlike most other packages, the derivatives are all computed analytically. Since the hyperparameters intervene in gradients one needs to compute the matrix derivatives via the kernel derivatives. If $K$ was defined via $k(x,x')$ then :

$$ \frac{d K}{d\theta}  = J_\theta$$

Where $J_\theta$ was defined via $\frac{dk(x,x')}{d\theta}$, the rest of the work is simply matrix algebra.

## !!! In construction !!!
