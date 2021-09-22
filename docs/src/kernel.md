# Kernels (Covariance functions)

Kernels are entirely defered to the package [KernelFunctions.jl](https://github.com/JuliaGaussianProcesses/KernelFunctions.jl), you can have a look at the documentation to see which are available. Note that, for now, optimization is only possible for `ScaleTransform` or `ARDTransform` with `ForwardDiff` while all others should be compatible with `Zygote`.

## Hyperparameter optimization

The advantage of Gaussian Processes is that it is possible to optimize all the hyperparameters of the model by optimizing the lower bound on the log evidence. One can compute the gradient of it and apply a classical gradient descent algorithm.

Unlike most other packages, the derivatives are computed analytically. One needs to compute the matrix derivatives via the kernel derivatives. If $K$ was defined via $k(x,x')$ then :

$$\frac{d K}{d\theta}  = J_\theta$$

Where $J_\theta$ was defined via $\frac{dk(x,x')}{d\theta}$.
This part is done by automatic differentiation. To chose between Zygote or ForwardDiff use `AGP.setKadbackend(:reverse_diff)` or `AGP.setKadbackend(:forward_diff)` respectively.

The rest of the work is simply matrix algebra.
