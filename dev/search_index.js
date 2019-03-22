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
    "text": "Théo Galy-Fajou PhD Student at Technical University of Berlin.\nFlorian Wenzel PhD Student at Technical University of Kaiserslautern and Humboldt University of Berlin"
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
    "text": "Here is a simple example to start right away :using AugmentedGaussianProcesses\nmodel = SVGP(X_train,y_train,kernel=RBFKernel(1.0),AugmentedLogisticLikelihood(),AnalyticInference(),m=50)\ntrain!(model,iterations=100)\ny_pred = predict_y(model,X_test)"
},

{
    "location": "#Related-Gaussian-Processes-packages-1",
    "page": "Home",
    "title": "Related Gaussian Processes packages",
    "category": "section",
    "text": "GaussianProcesses.jl : General package for Gaussian Processes with many available likelihoods\nStheno.jl : Package for Gaussian Process regression"
},

{
    "location": "#License-1",
    "page": "Home",
    "title": "License",
    "category": "section",
    "text": "AugmentedGaussianProcesses.jl is licensed under the MIT \"Expat\" license; see LICENSE for the full license text."
},

{
    "location": "background/#",
    "page": "Background",
    "title": "Background",
    "category": "page",
    "text": ""
},

{
    "location": "background/#The-bits-of-math-and-science-behind-it-1",
    "page": "Background",
    "title": "The bits of math and science behind it",
    "category": "section",
    "text": ""
},

{
    "location": "background/#Gaussian-Processes-1",
    "page": "Background",
    "title": "Gaussian Processes",
    "category": "section",
    "text": "To quote Wikipedia \"A Gaussian process is a stochastic process (a collection of random variables indexed by time or space), such that every finite collection of those random variables has a multivariate normal distribution, i.e. every finite linear combination of them is normally distributed. The distribution of a Gaussian process is the joint distribution of all those (infinitely many) random variables, and as such, it is a distribution over functions with a continuous domain, e.g. time or space.\"For a detailed understanding of Gaussian processes, check the wonderful book of Rasmussen and Williams and for a quick introduction, check this tutorial by Zoubin GhahramaniGaussian Processes are extremely practical model since they are non-parametric and are Bayesian. However the basic model is limited to regression with Gaussian noise and does not scale very well to large datasets (>1000 samples). The Augmented Gaussian Processes solve both these problems by adding inducing points as well as transforming the likelihood to get efficient variational inference."
},

{
    "location": "background/#Augmented-Gaussian-Processes-1",
    "page": "Background",
    "title": "Augmented Gaussian Processes",
    "category": "section",
    "text": "We are interested in models which consist of a GP prior on a latent function fsim textGP(0k), where k is the kernel function and the data y is connected to f via a non-conjugate likelihood p(yf) . We now aim on finding an augmented representation of the model which renders the model conditionally conjugate. Let omega be potential augmentation, then the augmented joint distribution isp(yfomega) =p(yfomega)p(omega)p(f).The original model can be restored by marginalizing omega, i.e. p(yf) =int p(yfomega)domega.The  goal  is  to  find  an  augmentation omega,  such  that  the  augmented  likelihood p(yfomega) becomes conjugate to the prior distributions p(f) and p(omega) and the expectations of the log complete conditional distributions log p(fomegay) and log p(omegafy) can be computed in closed-form."
},

{
    "location": "background/#How-to-find-a-suitable-augmentation?-1",
    "page": "Background",
    "title": "How to find a suitable augmentation?",
    "category": "section",
    "text": "Many popular likelihood functions can be expressed as ascale mixture of Gaussiansp(yf) =int N(yBftextdiag(omega^1))p(omega)domegawhere B is a matrix (Palmer et al., 2006).  This representation directly leads to the augmented likelihood p(yomegaf) =N(yBftextdiag(omega^1)) which is conjugate in f, i.e. the posterior is Gaussian again."
},

{
    "location": "background/#Inference-in-the-augmented-model-1",
    "page": "Background",
    "title": "Inference in the augmented model",
    "category": "section",
    "text": "We n assume that the augmentation, discussed in the previous section, was successful and we obtained an augmented model p(yfomega) = p(yfomega)p(f)p(omega) that is conditionally conjugate. In a conditionally conjugate model variational inference is easy and block coordinate ascent updates can be computed in closed-form. We follow as structured mean-field approach and assume a decoupling between the latent GP f and the auxiliary variable omega in the variational distribution q(fomega) = q(f) q(omega).  We alternate between updating q(f) and q(omega) by using the typical coordinate ascent (CAVI) updates building on expectations of the log complete conditionals.The hyperparameter of the latent GP (e.g. length scale) are learned by optimizing the variational lower bound as function of the hyper parameters. We alternate between updating the variational parameters and the hyperparameters."
},

{
    "location": "background/#Sparse-Gaussian-Processes-1",
    "page": "Background",
    "title": "Sparse Gaussian Processes",
    "category": "section",
    "text": "Direct inference for GPs has a cubic computational complexity mathcalO(N^3). To scale our model to big datasets we approximate the latent GP by a sparse GP building on inducing points. This reduces the complexity to mathcalO(M^3), where M is the number of inducing points. Using inducing points allows us to employ stochastic variational inference (SVI) that computes the updates based on mini-batches of the data."
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
    "location": "userguide/#init-1",
    "page": "User Guide",
    "title": "Initialization",
    "category": "section",
    "text": ""
},

{
    "location": "userguide/#GP-vs-VGP-vs-SVGP-1",
    "page": "User Guide",
    "title": "GP vs VGP vs SVGP",
    "category": "section",
    "text": "GP corresponds to the original GP regression model\nVGP is a variational GP model, a multivariate Gaussian is approximating the true posterior. There is no inducing points augmentation involved. Therefore it is fitted for small datasets (~10^3 samples)\nSVGP is a variational GP model augmented with inducing points. The optimization is done on those points, allowing for stochastic updates and huge scalability. The counterpart can be a slightly lower accuracy and the need to select the number and the location of the inducing points (however this is a problem currently worked on)."
},

{
    "location": "userguide/#Regression-1",
    "page": "User Guide",
    "title": "Regression",
    "category": "section",
    "text": "For regression one can use the Gaussian or StudentT likelihood. The first one is the vanilla GP with Gaussian noise while the second is using the Student-T likelihood and is therefore a lot more robust to ouliers."
},

{
    "location": "userguide/#Classification-1",
    "page": "User Guide",
    "title": "Classification",
    "category": "section",
    "text": "For classification one can select a Bernoulli likelihood with a logistic link or the BayesianSVM model based on the frequentist SVM."
},

{
    "location": "userguide/#Model-creation-1",
    "page": "User Guide",
    "title": "Model creation",
    "category": "section",
    "text": "Creating a model is as simple as doing VGP(X,y,kernel,likelihood,inference;args...) where args is described in the API. The compatibility of likelihoods and inferences is described in the next section and is regularly updated. For the kernels check out the kernel section"
},

{
    "location": "userguide/#Compatibility-table-1",
    "page": "User Guide",
    "title": "Compatibility table",
    "category": "section",
    "text": "Likelihood/Inference AnalyticInference GibbsSampling NumericalInference(Quadrature) NumericalInference(MCMCIntegration)\nGaussianLikelihood ✔ ✖ ✖ ✖\nStudentTLikelihood ✔ (dev) (dev) ✖\nLogisticLikelihood ✔ (dev) (dev) ✖\nBayesianSVM ✔ ✖ (dev) ✖\nLogisticSoftMaxLikelihood ✔ (dev) ✖ (dev)\nSoftMaxLikelihood ✖ ✖ ✖ (dev)"
},

{
    "location": "userguide/#train-1",
    "page": "User Guide",
    "title": "Training",
    "category": "section",
    "text": "Training is straightforward after initializing the model by running :train!(model;iterations=100,callback=callbackfunction)Where the callback option is for running a function at every iteration. callback function should be defined asfunction callbackfunction(model,iter)\n    \"do things here\"...\nend"
},

{
    "location": "userguide/#pred-1",
    "page": "User Guide",
    "title": "Prediction",
    "category": "section",
    "text": "Once the model has been trained it is finally possible to compute predictions. There always three possibilities :predict_f(model,X_test,covf=true,fullcov=false) : Compute the parameters (mean and covariance) of the latent normal distributions of each test points. If covf=false return only the mean, if fullcov=true return a covariance matrix instead of only the diagonal\npredict_y(model,X_test) : Compute the point estimate of the predictive likelihood for regression or the label of the most likely class for classification.\nproba_y(model,X_test) : Return the mean with the variance of eahc point for regression or the predictive likelihood to obtain the class y=1 for classification."
},

{
    "location": "userguide/#Miscellaneous-1",
    "page": "User Guide",
    "title": "Miscellaneous",
    "category": "section",
    "text": "In construction <!– ### Saving/Loading modelsOnce a model has been trained it is possible to save its state in a file by using  save_trained_model(filename,model), a partial version of the file will be save in filename.It is then possible to reload this file by using load_trained_model(filename). !!!However note that it will not be possible to train the model further!!! This function is only meant to do further predictions."
},

{
    "location": "userguide/#Pre-made-callback-functions-1",
    "page": "User Guide",
    "title": "Pre-made callback functions",
    "category": "section",
    "text": "There is one (for now) premade function to return a a MVHistory object and callback function for the training of binary classification problems. The callback will store the ELBO and the variational parameters at every iterations included in iterpoints If Xtest and y_test are provided it will also store the test accuracy and the mean and median test loglikelihood –>"
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
    "text": "The kernel function or covariance function is a crucial part of Gaussian Processes. It determines the covariance matrices between set of points, and its behaviour and parameters determines almost completely a GP behaviour."
},

{
    "location": "kernel/#Kernels-available-1",
    "page": "Kernels",
    "title": "Kernels available",
    "category": "section",
    "text": "To get a short introduction of some covariance functions available one can look at these Slides from RasmussenWe define theta_i as the lengthscale (in case of IsoKernel theta_i=thetaforall i) and sigma is the variance In this package covariance functions are progressively added for now the available kernels are :RBF Kernel or Squared Exponential Kernelk(xx) = sigma expleft(-frac12sum_i=1^Dfrac(x_i-x_i)^2theta_i^2right)Matern 3/2 Kernelk(xx)= sigmaleft(1+sqrt3sum frac(x_i - x_i)^2theta_i^2right)expleft(-sqrt3sum frac(x_i - x_i)^2theta_i^2right)More are coming, check the github projects for updates ."
},

{
    "location": "kernel/#Hyperparameter-optimization-1",
    "page": "Kernels",
    "title": "Hyperparameter optimization",
    "category": "section",
    "text": "The advantage of Gaussian Processes is that it is possible to optimize all the hyperparameters of the model by optimizing the lower bound on the loglikelihood. One can compute the gradient of it and apply a classical gradient descent algorithm.Unlike most other packages, the derivatives are all computed analytically. Since the hyperparameters intervene in gradients one needs to compute the matrix derivatives via the kernel derivatives. If K was defined via k(xx) then :$ \\frac{d K}{d\\theta}  = J_\\theta$Where J_theta was defined via fracdk(xx)dtheta, the rest of the work is simply matrix algebra."
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
    "location": "api/#Model-Types-1",
    "page": "API",
    "title": "Model Types",
    "category": "section",
    "text": "GP(X,y,kernel)\nVGP(X,y,kernel,likelihood,inference)\nSVGP(X,y,kernel,likelihood,inference,num_ind_points)"
},

{
    "location": "api/#Likelihood-Types-1",
    "page": "API",
    "title": "Likelihood Types",
    "category": "section",
    "text": "GaussianLikelihood()\nAugmentedStudentTLikelihood(ν)\nBayesianSVM()\nAugmentedLogisticLikelihood()\nAugmentedLogisticSoftMaxLikelihood()"
},

{
    "location": "api/#Inference-Types-1",
    "page": "API",
    "title": "Inference Types",
    "category": "section",
    "text": "AnalyticInference()\nStochasticAnalyticInference(batchsize)\nNumericalInference(optimizer)\nStochasticNumericalInference(batchsize)\nGibbsSampling()"
},

{
    "location": "api/#Functions-and-methods-1",
    "page": "API",
    "title": "Functions and methods",
    "category": "section",
    "text": "train!(model)\npredict_f(model,X_test)\npredict_y(model,X_test)\nproba_y(model,X_test)"
},

{
    "location": "api/#Kernels-1",
    "page": "API",
    "title": "Kernels",
    "category": "section",
    "text": "RBFKernel()\nMaternKernel)_"
},

{
    "location": "api/#AugmentedGaussianProcesses.KernelModule.kernelmatrix",
    "page": "API",
    "title": "AugmentedGaussianProcesses.KernelModule.kernelmatrix",
    "category": "function",
    "text": "Create the covariance matrix between the matrix X1 and X2 with the covariance function kernel\n\n\n\n\n\nCompute the covariance matrix of the matrix X, optionally only compute the diagonal terms\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.KernelModule.kernelmatrix!",
    "page": "API",
    "title": "AugmentedGaussianProcesses.KernelModule.kernelmatrix!",
    "category": "function",
    "text": "Compute the covariance matrix between the matrix X1 and X2 with the covariance function kernel in preallocated matrix K\n\n\n\n\n\nCompute the covariance matrix of the matrix X in preallocated matrix K, optionally only compute the diagonal terms\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.KernelModule.getvariance",
    "page": "API",
    "title": "AugmentedGaussianProcesses.KernelModule.getvariance",
    "category": "function",
    "text": "Return the variance of the kernel\n\n\n\n\n\n"
},

{
    "location": "api/#AugmentedGaussianProcesses.KernelModule.getlengthscales",
    "page": "API",
    "title": "AugmentedGaussianProcesses.KernelModule.getlengthscales",
    "category": "function",
    "text": "Return the lengthscale of the IsoKernel\n\n\n\n\n\nReturn the lengthscales of the ARD Kernel\n\n\n\n\n\n"
},

{
    "location": "api/#Kernel-functions-1",
    "page": "API",
    "title": "Kernel functions",
    "category": "section",
    "text": "kernelmatrix\nkernelmatrix!\ngetvariance\ngetlengthscales"
},

{
    "location": "api/#Index-1",
    "page": "API",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"api.md\"]\nModule = [\"AugmentedGaussianProcesses\"]\nOrder = [:type, :function]"
},

]}
