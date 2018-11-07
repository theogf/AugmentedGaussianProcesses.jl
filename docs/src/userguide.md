# User Guide

There are 3 main stages for the GPs:

- Initialization
- Training
- Prediction

## Initialization

### Sparse vs FullBatch

- A `FullBatchModel` is a normal GP model, where the variational distribution is optimized over all the training points and no stochastic updates are possible. It is therefore fitted for small datasets (~10^3 samples)
- A `SparseModel` is a GP model augmented with inducing points. The optimization is done on those points, allowing for stochastic updates and huge scalability. The counterpart can be a slightly lower accuracy and the need to select the number and the location of the inducing points (however this is a problem currently worked on).

To pick between the two add `Batch` or `Sparse` in front of the name of the desired model.

### Regression

For **regression** one can use the `GPRegression` or `StudentT` model. The latter is using the __Student-T__ likelihood and is therefore a lot more robust to ouliers.

### Classification

For **classification** one can use the `XGPC` using the [logistic link](https://en.wikipedia.org/wiki/Logistic_function) or the `BSVM` model based on the [frequentist SVM](https://en.wikipedia.org/wiki/Support_vector_machine#Bayesian_SVM).


### Model creation

Creating a model is as simple as doing `GPModel(X,y;args...)` where `args` is described in the next section

### Parameters of the models

All models except for `BatchGPRegression` use the same set of parameters for initialisation. Default values are showed as well.

One of the main parameter is the kernel function (or covariance function). This detailed in [the kernel section](https://theogf.github.io/AugmentedGaussianProcesses.jl/latest/Kernels), by default an isotropic RBFKernel with lengthscale 1.0 is used.

Common parameters :

- `Autotuning::Bool=true` : Are the hyperparameters trained
- `nEpochs::Integer=100` : How many iteration steps are used (can also be set at training time)
- kernel::Kernel : Kernel for the model
- `noise::Real=1e-3` : noise added in the model (is also optimized)
- `ϵ::Real=1e-5` : minimum value for convergence
- `SmoothingWindow::Integer=5` : Window size for averaging convergence in the stochastic case
- `verbose::Integer=0` : How much information is displayed (from 0 to 3)

Specific to sparse models :


- `m::Integer=0` : Number of inducing points (if not given will be set to a default value depending of the number of points)
- `Stochastic::Bool=true` : Is the method trained via mini batches
- `batchsize::Integer=-1` : number of samples per minibatches (must be set to a correct value or model will fall back for a non stochastic inference)
- `AdaptiveLearningRate::Bool=true` : Is the learning rate adapted via estimation of the gradient variance? see ["Adaptive Learning Rate for Stochastic Variational inference"](https://pdfs.semanticscholar.org/9903/e08557f328d58e4ba7fce68faee380d30b12.pdf), if not use simple exponential decay with parameters `κ_s` and `τ_s` seen under `(1/(iter+τ_s))^-κ_s`
- `κ_s::Real=1.0`
- `τ_s::Real=100`
- `optimizer::Optimizer=Adam()` : Type of optimizer for the inducing point locations
- `OptimizeIndPoints::Bool=false` : Is the location of inducing points optimized, the default is currently to false, as the method is relatively inefficient and does not bring much better performances

Model specific :
`StudentT` : `ν::Real=5` : Number of degrees of freedom of the Student-T likelihood

## Training

Training is straightforward after initializing the model by running :
```julia
model.train(;iterations=100,callback=callbackfunction)
```
Where the `callback` option is for running a function at every iteration. `callback function should be defined as`
```julia
function callbackfunction(model,iter)
    "do things here"...
end
```

## Prediction

Once the model has been trained it is finally possible to compute predictions. There always three possibilities :

- `model.fstar(X_test,covf=true)` : Compute the parameters (mean and covariance) of the latent normal distributions of each test points. If `covf=false` return only the mean.
- `model.predict(X_test)` : Compute the point estimate of the predictive likelihood for regression or the label of the most likely class for classification.
- `model.predictproba(X_test)` : Compute the exact predictive likelihood for regression or the predictive likelihood to obtain the class `y=1` for classification.

## Miscellaneous

### Saving/Loading models

Once a model has been trained it is possible to save its state in a file by using  `save_trained_model(filename,model)`, a partial version of the file will be save in `filename`.

It is then possible to reload this file by using `load_trained_model(filename)`. !!!However note that it will not be possible to train the model further!!! This function is only meant to do further predictions.

### Pre-made callback functions

There is one (for now) premade function to return a a MVHistory object and callback function for the training of binary classification problems.
The callback will store the ELBO and the variational parameters at every iterations included in iter_points
If X_test and y_test are provided it will also store the test accuracy and the mean and median test loglikelihood
