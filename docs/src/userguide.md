# User Guide

There are 3 main stages for the GPs:

- Initialization
- Training
- Prediction

## Initialization

One must first pick the model fitting the best its needs. The first criteria is the size of the datasets: if your datasets contains around 1000 samples or less it is better to use a `FullBatchModel` which is the most accurate (but also sometimes more prone to overfitting).
If there is more, it is worth using a `SparseModel` based on inducing points.

### Sparse vs FullBatch

- A `FullBatchModel` is a normal GP model, where the variational distribution is optimized over all the training points and no stochastic updates are possible. It is therefore fitted for small datasets (~10^3 samples)
- A `SparseModel` is a GP model augmented with inducing points. The optimization is done on those points, allowing for stochastic updates and huge scalability. The counterpart can be a slightly lower accuracy and the need to select the number and the location of the inducing points (however this is a problem currently worked on).

To pick between the two add `Batch` or `Sparse` in front of the name of the desired model.

### Regression

For **regression** one can use the `GPRegression` or `StudentT` model. The latter is using the __Student-T__ likelihood and is therefore a lot more robust to ouliers.

### Classification

For **classification** one can use the `XGPC` using the [logistic link](https://en.wikipedia.org/wiki/Logistic_function) or the `BSVM` model based on the [frequentist SVM](https://en.wikipedia.org/wiki/Support_vector_machine#Bayesian_SVM).

### Parameters of the models

All models except for `BatchGPRegression` use the same set of parameters for initialisation. Default values are showed as well

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
- `AdaptiveLearningRate::Bool=true` : Is the learning rate adapted via estimation of the gradient variance? see ["Adaptive Learning Rate for Stochastic Variational inference"]( https://pdfs.semanticscholar.org/9903/e08557f328d58e4ba7fce68faee380d30b12.pdf), if not use simple exponential decay with parameters `κ_s` and `τ_s` seen under `(1/(iter+τ_s))^-κ_s`
- `κ_s::Real=1.0`
- `τ_s::Real=100`
- `optimizer::Optimizer=Adam()` : Type of optimizer for the inducing point locations
- `OptimizeIndPoints::Bool=false` : Is the location of inducing points optimized, the default is currently to false, as the method is relatively inefficient and does not bring much better performances

Model specific :
`StudentT` : `ν::Real=5` : Number of degrees of freedom of the Student-T likelihood

## Training



## Prediction
