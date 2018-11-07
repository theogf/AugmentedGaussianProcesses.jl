var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#AugmentedGaussianProcesses.jl-1",
    "page": "Home",
    "title": "AugmentedGaussianProcesses.jl",
    "category": "section",
    "text": "(Image: ) A Julia package for Augmented and Normal Gaussian Processes."
},

{
    "location": "#Authors-1",
    "page": "Home",
    "title": "Authors",
    "category": "section",
    "text": "Theo Galy-Fajou PhD Student at Technical University of Berlin.\nFlorian Wenzel PhD Student at Technical University of Kaiserslautern"
},

{
    "location": "#License-1",
    "page": "Home",
    "title": "License",
    "category": "section",
    "text": "AugmentedGaussianProcesses.jl is licensed under the MIT \"Expat\" license; see LICENSE for the full license text."
},

{
    "location": "#Installation-1",
    "page": "Home",
    "title": "Installation",
    "category": "section",
    "text": "AugmentedGaussianProcesses is a registered package and is symply installed by runningpkg> add AugmentedGaussianProcesses"
},

{
    "location": "#Basic-example-1",
    "page": "Home",
    "title": "Basic example",
    "category": "section",
    "text": "Here is a simple example to start right away :using AugmentedGaussianProcesses\nmodel = SparseXGPC(X_train,y_train,kernel=RBFKernel(1.0),m=50)\nmodel.train(iterations=100)\ny_pred = model.predict(X_test)"
},

{
    "location": "#Related-Gaussian-Processes-packages-1",
    "page": "Home",
    "title": "Related Gaussian Processes packages",
    "category": "section",
    "text": "GaussianProcesses.jl : General package for Gaussian Processes with many available likelihoods\nStheno.jl : Package for Gaussian Process regression"
},

{
    "location": "background/#",
    "page": "Background",
    "title": "Background",
    "category": "page",
    "text": ""
},

{
    "location": "background/#The-bits-of-science-behing-it-1",
    "page": "Background",
    "title": "The bits of science behing it",
    "category": "section",
    "text": ""
},

{
    "location": "background/#!!!-In-construction-!!!-1",
    "page": "Background",
    "title": "!!! In construction !!!",
    "category": "section",
    "text": ""
},

{
    "location": "background/#Gaussian-Processes-1",
    "page": "Background",
    "title": "Gaussian Processes",
    "category": "section",
    "text": ""
},

{
    "location": "background/#Augmented-Gaussian-Processes-1",
    "page": "Background",
    "title": "Augmented Gaussian Processes",
    "category": "section",
    "text": ""
},

{
    "location": "background/#Sparse-Gaussian-Processes-1",
    "page": "Background",
    "title": "Sparse Gaussian Processes",
    "category": "section",
    "text": ""
},

{
    "location": "userguide/#",
    "page": "User Guide",
    "title": "User Guide",
    "category": "page",
    "text": ""
},

{
    "location": "userguide/#User-Guide-1",
    "page": "User Guide",
    "title": "User Guide",
    "category": "section",
    "text": "There are 3 main stages for the GPs:Initialization\nTraining\nPrediction"
},

{
    "location": "userguide/#Initialization-1",
    "page": "User Guide",
    "title": "Initialization",
    "category": "section",
    "text": ""
},

{
    "location": "userguide/#Sparse-vs-FullBatch-1",
    "page": "User Guide",
    "title": "Sparse vs FullBatch",
    "category": "section",
    "text": "A FullBatchModel is a normal GP model, where the variational distribution is optimized over all the training points and no stochastic updates are possible. It is therefore fitted for small datasets (~10^3 samples)\nA SparseModel is a GP model augmented with inducing points. The optimization is done on those points, allowing for stochastic updates and huge scalability. The counterpart can be a slightly lower accuracy and the need to select the number and the location of the inducing points (however this is a problem currently worked on).To pick between the two add Batch or Sparse in front of the name of the desired model."
},

{
    "location": "userguide/#Regression-1",
    "page": "User Guide",
    "title": "Regression",
    "category": "section",
    "text": "For regression one can use the GPRegression or StudentT model. The latter is using the Student-T likelihood and is therefore a lot more robust to ouliers."
},

{
    "location": "userguide/#Classification-1",
    "page": "User Guide",
    "title": "Classification",
    "category": "section",
    "text": "For classification one can use the XGPC using the logistic link or the BSVM model based on the frequentist SVM."
},

{
    "location": "userguide/#Model-creation-1",
    "page": "User Guide",
    "title": "Model creation",
    "category": "section",
    "text": "Creating a model is as simple as doing GPModel(X,y;args...) where args is described in the next section"
},

{
    "location": "userguide/#Parameters-of-the-models-1",
    "page": "User Guide",
    "title": "Parameters of the models",
    "category": "section",
    "text": "All models except for BatchGPRegression use the same set of parameters for initialisation. Default values are showed as well.One of the main parameter is the kernel function (or covariance function). This detailed in the kernel section, by default an isotropic RBFKernel with lengthscale 1.0 is used.Common parameters :Autotuning::Bool=true : Are the hyperparameters trained\nnEpochs::Integer=100 : How many iteration steps are used (can also be set at training time)\nkernel::Kernel : Kernel for the model\nnoise::Real=1e-3 : noise added in the model (is also optimized)\nϵ::Real=1e-5 : minimum value for convergence\nSmoothingWindow::Integer=5 : Window size for averaging convergence in the stochastic case\nverbose::Integer=0 : How much information is displayed (from 0 to 3)Specific to sparse models :m::Integer=0 : Number of inducing points (if not given will be set to a default value depending of the number of points)\nStochastic::Bool=true : Is the method trained via mini batches\nbatchsize::Integer=-1 : number of samples per minibatches (must be set to a correct value or model will fall back for a non stochastic inference)\nAdaptiveLearningRate::Bool=true : Is the learning rate adapted via estimation of the gradient variance? see \"Adaptive Learning Rate for Stochastic Variational inference\", if not use simple exponential decay with parameters κ_s and τ_s seen under (1/(iter+τ_s))^-κ_s\nκ_s::Real=1.0\nτ_s::Real=100\noptimizer::Optimizer=Adam() : Type of optimizer for the inducing point locations\nOptimizeIndPoints::Bool=false : Is the location of inducing points optimized, the default is currently to false, as the method is relatively inefficient and does not bring much better performancesModel specific : StudentT : ν::Real=5 : Number of degrees of freedom of the Student-T likelihood"
},

{
    "location": "userguide/#Training-1",
    "page": "User Guide",
    "title": "Training",
    "category": "section",
    "text": "Training is straightforward after initializing the model by running :model.train(;iterations=100,callback=callbackfunction)Where the callback option is for running a function at every iteration. callback function should be defined asfunction callbackfunction(model,iter)\n    \"do things here\"...\nend"
},

{
    "location": "userguide/#Prediction-1",
    "page": "User Guide",
    "title": "Prediction",
    "category": "section",
    "text": "Once the model has been trained it is finally possible to compute predictions. There always three possibilities :model.fstar(X_test,covf=true) : Compute the parameters (mean and covariance) of the latent normal distributions of each test points. If covf=false return only the mean.\nmodel.predict(X_test) : Compute the point estimate of the predictive likelihood for regression or the label of the most likely class for classification.\nmodel.predictproba(X_test) : Compute the exact predictive likelihood for regression or the predictive likelihood to obtain the class y=1 for classification."
},

{
    "location": "userguide/#Miscellaneous-1",
    "page": "User Guide",
    "title": "Miscellaneous",
    "category": "section",
    "text": ""
},

{
    "location": "userguide/#Saving/Loading-models-1",
    "page": "User Guide",
    "title": "Saving/Loading models",
    "category": "section",
    "text": "Once a model has been trained it is possible to save its state in a file by using  save_trained_model(filename,model), a partial version of the file will be save in filename.It is then possible to reload this file by using load_trained_model(filename). !!!However note that it will not be possible to train the model further!!! This function is only meant to do further predictions."
},

{
    "location": "userguide/#Pre-made-callback-functions-1",
    "page": "User Guide",
    "title": "Pre-made callback functions",
    "category": "section",
    "text": "There is one (for now) premade function to return a a MVHistory object and callback function for the training of binary classification problems. The callback will store the ELBO and the variational parameters at every iterations included in iterpoints If Xtest and y_test are provided it will also store the test accuracy and the mean and median test loglikelihood"
},

{
    "location": "kernel/#",
    "page": "Kernels",
    "title": "Kernels",
    "category": "page",
    "text": ""
},

{
    "location": "kernel/#Kernels-(Covariance-functions)-1",
    "page": "Kernels",
    "title": "Kernels (Covariance functions)",
    "category": "section",
    "text": ""
},

{
    "location": "kernel/#!!!-In-construction-!!!-1",
    "page": "Kernels",
    "title": "!!! In construction !!!",
    "category": "section",
    "text": ""
},

{
    "location": "examples/#",
    "page": "Examples",
    "title": "Examples",
    "category": "page",
    "text": ""
},

{
    "location": "examples/#Examples-1",
    "page": "Examples",
    "title": "Examples",
    "category": "section",
    "text": "The best way to understand how the package is working is to look at examples. For each model you can find a Jupyter notebook on this repository"
},

{
    "location": "api/#",
    "page": "API",
    "title": "API",
    "category": "page",
    "text": ""
},

{
    "location": "api/#Library-1",
    "page": "API",
    "title": "Library",
    "category": "section",
    "text": "CurrentModule = AugmentedGaussianProcesses"
},

{
    "location": "api/#AugmentedGaussianProcesses.AugmentedGaussianProcesses",
    "page": "API",
    "title": "AugmentedGaussianProcesses.AugmentedGaussianProcesses",
    "category": "module",
    "text": "General Framework for the data augmented Gaussian Processes\n\n\n\n\n\n"
},

{
    "location": "api/#Module-1",
    "page": "API",
    "title": "Module",
    "category": "section",
    "text": "AugmentedGaussianProcesses"
},

{
    "location": "api/#AugmentedGaussianProcesses.BatchGPRegression",
    "page": "API",
    "title": "AugmentedGaussianProcesses.BatchGPRegression",
    "category": "type",
    "text": "Classic Batch Gaussian Process Regression (no inducing points)\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.SparseGPRegression",
    "page": "API",
    "title": "AugmentedGaussianProcesses.SparseGPRegression",
    "category": "type",
    "text": "Sparse Gaussian Process Regression with Gaussian Likelihood\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.BatchStudentT",
    "page": "API",
    "title": "AugmentedGaussianProcesses.BatchStudentT",
    "category": "type",
    "text": "Batch Student T Gaussian Process Regression (no inducing points)\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.SparseStudentT",
    "page": "API",
    "title": "AugmentedGaussianProcesses.SparseStudentT",
    "category": "type",
    "text": "Sparse Gaussian Process Regression with Student T likelihood\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.BatchXGPC",
    "page": "API",
    "title": "AugmentedGaussianProcesses.BatchXGPC",
    "category": "type",
    "text": "Batch Gaussian Process Classifier with Logistic Likelihood (no inducing points)\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.SparseXGPC",
    "page": "API",
    "title": "AugmentedGaussianProcesses.SparseXGPC",
    "category": "type",
    "text": "Sparse Gaussian Process Classifier with Logistic Likelihood Create a GP model taking the  training data and labels X & y as required arguments. Other optional arguments are:\n\nStochastic::Bool : Is the method trained via mini batches\nAdaptiveLearningRate::Bool : Is the learning rate adapted via estimation of the gradient variance? see \"Adaptive Learning Rate for Stochastic Variational inference\" https://pdfs.semanticscholar.org/9903/e08557f328d58e4ba7fce68faee380d30b12.pdf, if not use simple exponential decay with parameters κs and τs seen under (1/(iter+τs))^-κs\nAutotuning::Bool : Are the hyperparameters trained as well\noptimizer::Optimizer : Type of optimizer for the hyperparameters\nOptimizeIndPoints::Bool : Is the location of inducing points optimized\nnEpochs::Integer : How many iteration steps\nbatchsize::Integer : number of samples per minibatches\nκ_s::Real\nτ_s::Real\nkernel::Kernel : Kernel for the model\nnoise::Float64 : noise added in the model\nm::Integer : Number of inducing points\nϵ::Float64 : minimum value for convergence\nSmoothingWindow::Integer : Window size for averaging convergence in the stochastic case\nverbose::Integer : How much information is displayed (from 0 to 3)\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.BatchBSVM",
    "page": "API",
    "title": "AugmentedGaussianProcesses.BatchBSVM",
    "category": "type",
    "text": "Batch Bayesian Support Vector Machine (no inducing points)\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.SparseBSVM",
    "page": "API",
    "title": "AugmentedGaussianProcesses.SparseBSVM",
    "category": "type",
    "text": "Sparse Gaussian Process Classifier with Bayesian SVM likelihood\n\n\n\n\n\n"
},

{
    "location": "api/#Types-1",
    "page": "API",
    "title": "Types",
    "category": "section",
    "text": "BatchGPRegression\nSparseGPRegression\nBatchStudentT\nSparseStudentT\nBatchXGPC\nSparseXGPC\nBatchBSVM\nSparseBSVM"
},

{
    "location": "api/#AugmentedGaussianProcesses.train!",
    "page": "API",
    "title": "AugmentedGaussianProcesses.train!",
    "category": "function",
    "text": "Function to train the given Online GP model, there are options to change the number of max iterations, give a callback function that will take the model and the actual step as arguments and give a convergence method to stop the algorithm given specific criteria\n\n\n\n\n\nFunction to train the given GP model, there are options to change the number of max iterations, give a callback function that will take the model and the actual step as arguments and give a convergence method to stop the algorithm given specific criteria\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.regpredict",
    "page": "API",
    "title": "AugmentedGaussianProcesses.regpredict",
    "category": "function",
    "text": "Return the mean of the predictive distribution of f\n\n\n\n\n\nReturn the mean of the predictive distribution of f\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.regpredictproba",
    "page": "API",
    "title": "AugmentedGaussianProcesses.regpredictproba",
    "category": "function",
    "text": "Return the mean and variance of the predictive distribution of f\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.studenttpredict",
    "page": "API",
    "title": "AugmentedGaussianProcesses.studenttpredict",
    "category": "function",
    "text": "Return the mean of the predictive distribution of f\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.studenttpredictproba",
    "page": "API",
    "title": "AugmentedGaussianProcesses.studenttpredictproba",
    "category": "function",
    "text": "Return the mean and variance of the predictive distribution of f\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.logitpredict",
    "page": "API",
    "title": "AugmentedGaussianProcesses.logitpredict",
    "category": "function",
    "text": "Return the predicted class {-1,1} with a GP model via the logit link\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.logitpredictproba",
    "page": "API",
    "title": "AugmentedGaussianProcesses.logitpredictproba",
    "category": "function",
    "text": "Return the mean of likelihood p(y=1|X,x) via the logit link with a GP model\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.svmpredict",
    "page": "API",
    "title": "AugmentedGaussianProcesses.svmpredict",
    "category": "function",
    "text": "Return the point estimate of the likelihood of class y=1 via the SVM likelihood\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.svmpredictproba",
    "page": "API",
    "title": "AugmentedGaussianProcesses.svmpredictproba",
    "category": "function",
    "text": "Return the likelihood of class y=1 via the SVM likelihood\n\n\n\n\n\n"
},

{
    "location": "api/#Functions-and-methods-1",
    "page": "API",
    "title": "Functions and methods",
    "category": "section",
    "text": "train!\nregpredict\nregpredictproba\nstudenttpredict\nstudenttpredictproba\nlogitpredict\nlogitpredictproba\nsvmpredict\nsvmpredictproba"
},

{
    "location": "api/#Internals-1",
    "page": "API",
    "title": "Internals",
    "category": "section",
    "text": ""
},

{
    "location": "api/#Index-1",
    "page": "API",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"api.md\"]\nModule = [\"AugmentedGaussianProcesses\"]\nOrder = [:type, :function]"
},

]}
