#Simple tool to define macros
macro def(name, definition)
    return quote
        macro $(esc(name))()
            esc($(Expr(:quote, definition)))
        end
    end
end



@def commonfields begin
    #Data
    X::Matrix{Float64} #Feature vectors
    y::Vector #Output (-1,1 for classification, real for regression, matrix for multiclass)
    ModelType::GPModelType; #Type of model
    Name::String #Name of the model
    nSamples::Int64 # Number of data points
    nDim::Int64
    nFeatures::Int64 # Number of features
    noise::KernelModule.HyperParameter{Float64}  #Regularization parameter of the noise
    ϵ::Float64  #Desired Precision on ||ELBO(t+1)-ELBO(t)||))
    evol_conv::Vector{Float64} #Used for convergence estimation
    prev_params::Any
    nEpochs::Int64; #Maximum number of iterations
    verbose::Int64 #Level of printing information
    Stochastic::Bool #Is the model stochastic    #Autotuning parameters
    Autotuning::Bool #Chose Autotuning type
    AutotuningFrequency::Int64 #Frequency of update of the hyperparameter
    Trained::Bool #Verify the algorithm has been trained before making predictions
    #Parameters learned with training
    TopMatrixForPrediction #Storing matrices for repeated predictions (top and down are numerator and discriminator)
    DownMatrixForPrediction
    MatricesPrecomputed::Bool #Flag to know if matrices needed for predictions are already computed or not
    HyperParametersUpdated::Bool #To know if the inverse kernel matrix must updated
end
"""
    initCommon!(model,...)

Initialize all the common fields of the different models, i.e. the dataset inputs `X` and outputs `y`,
the noise `noise`, the convergence threshold `ϵ`, the initial number of iterations `nEpochs`,
the `verboseLevel` (from 0 to 3), enabling `Autotuning`, the `AutotuningFrequency` and
what `optimizer` to use
"""
function initCommon!(model::GPModel,X::Matrix{Float64},y::Vector,noise::Float64,ϵ::Float64,nEpochs::Integer,verbose::Integer,Autotuning::Bool,AutotuningFrequency::Integer,optimizer::Optimizer)
    @assert (size(y,1)==size(X,1)) "There is a dimension problem with the data size(y)!=size(X)";
    model.X = X; model.y = y;
    @assert noise > 0 "noise should be a positive float";  model.noise = KernelModule.HyperParameter{Float64}(noise,KernelModule.interval(KernelModule.OpenBound{Float64}(zero(Float64)),KernelModule.NullBound{Float64}()))
    @assert ϵ > 0 "ϵ should be a positive float"; model.ϵ = ϵ;
    @assert nEpochs > 0 "nEpochs should be positive"; model.nEpochs = nEpochs;
    @assert (verbose > -1 && verbose < 4) "verbose should be in {0,1,2,3}, here value is $verbose"; model.verbose = verbose;
    model.Autotuning = Autotuning; model.AutotuningFrequency = AutotuningFrequency;
    # model.opt_type = optimizer;
    model.nSamples = size(X,1); #model.nSamplesUsed = model.nSamples;
    model.nDim= size(X,2);
    model.Trained = false; model.Stochastic = false;
    model.TopMatrixForPrediction = 0; model.DownMatrixForPrediction = 0; model.MatricesPrecomputed=false;
    model.HyperParametersUpdated = true;
    model.evol_conv = Vector()
end

"""
    Parameters for stochastic optimization
"""
@def stochasticfields begin
    nSamplesUsed::Int64 #Size of the minibatch used
    StochCoeff::Float64 #Stochastic Coefficient
    MBIndices #MiniBatch Indices
    #Flag for adaptative learning rate for the SVI
    AdaptiveLearningRate::Bool
      κ_s::Float64 #Parameters for decay of learning rate (iter + κ)^-τ in case adaptative learning rate is not used
      τ_s::Float64
    ρ_s::Float64 #Learning rate for CAVI
    g::Vector{Float64} # g & h are expected gradient value for computing the adaptive learning rate and τ is an intermediate
    h::Float64
    τ::Float64
    SmoothingWindow::Int64
end
"""
    Function initializing the stochasticfields parameters
"""
function initStochastic!(model::GPModel,AdaptiveLearningRate::Bool,batchsize::Integer,κ_s::Real,τ_s::Real,SmoothingWindow::Integer)
    #Initialize parameters specific to models using SVI and check for consistency
    model.Stochastic = true; model.nSamplesUsed = batchsize; model.AdaptiveLearningRate = AdaptiveLearningRate;
    model.κ_s = κ_s; model.τ_s = τ_s; model.SmoothingWindow = SmoothingWindow;
    if (model.nSamplesUsed <= 0 || model.nSamplesUsed > model.nSamples)
################### TODO MUST DECIDE FOR DEFAULT VALUE OR STOPPING STOCHASTICITY ######
        @warn "Invalid value for the batchsize : $batchsize, assuming a full batch method"
        model.nSamplesUsed = model.nSamples; model.Stochastic = false;
    end
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    model.τ = 50;
end


"""
    Parameters for the kernel parameters, including the covariance matrix of the prior
"""
@def kernelfields begin
    kernel::Kernel #Kernels function used
    Knn::Matrix{Float64} #Kernel matrix of the GP prior
    invK::Matrix{Float64} #Inverse Kernel Matrix for the nonlinear case
end
"""
Function initializing the kernelfields
"""
function initKernel!(model::GPModel,kernel::Kernel)
    #Initialize parameters common to all models containing kernels and check for consistency
    if kernel == 0
      @warn "No kernel indicated, a rbf kernel function with lengthscale 1 is used"
      kernel = RBFKernel(1.0)
    end
    model.kernel = deepcopy(kernel)
    model.nFeatures = model.nSamples
end
"""
    Parameters necessary for the sparse inducing points method
"""
@def sparsefields begin
    m::Int64 #Number of inducing points
    inducingPoints::Matrix{Float64} #Inducing points coordinates for the Big Data GP
    OptimizeInducingPoints::Bool #Flag for optimizing the points during training
    optimizer::Optimizer #Optimizer for the inducing points
    Kmm::Symmetric{Float64,Matrix{Float64}} #Kernel matrix
    invKmm::Symmetric{Float64,Matrix{Float64}} #Inverse Kernel matrix of inducing points
    Ktilde::Vector{Float64} #Diagonal of the covariance matrix between inducing points and generative points
    κ::Matrix{Float64} #Kmn*invKmm
    Knm::Matrix{Float64}
end
"""
Function initializing the sparsefields parameters
"""
function initSparse!(model::GPModel,m,optimizeIndPoints)
    #Initialize parameters for the sparse model and check consistency
    minpoints = 64;
    if m > model.nSamples
        @warn "There are more inducing points than actual points, setting it to min($minpoints,10%)"
        m = min(minpoints,model.nSamples÷10)
    elseif m == 0
        @warn "Number of inducing points was not manually set, setting it to min($minpoints,10%)"
        m = min(minpoints,model.nSamples÷10)
    end
    model.m = m; model.nFeatures = model.m;
    model.OptimizeInducingPoints = optimizeIndPoints
    model.optimizer = Adam(α=0.1);
    model.inducingPoints = KMeansInducingPoints(model.X,model.m,10)
    if model.verbose>1
        println("Inducing points determined through KMeans algorithm")
    end
    model.Kmm = Symmetric(Matrix{Float64}(undef,model.m,model.m)) #Kernel matrix
    model.invKmm = Symmetric(Matrix{Float64}(undef,model.m,model.m)) #Inverse Kernel matrix of inducing points
    model.Ktilde = Vector{Float64}(undef,model.m) #Diagonal of the covariance matrix between inducing points and generative points
    model.κ = Matrix{Float64}(undef,model.nSamplesUsed,model.m) #Kmn*invKmm
    model.Knm = Matrix{Float64}(undef,model.nSamplesUsed,model.m)
    # model.inducingPoints += rand(Normal(0,0.1),size(model.inducingPoints)...)
end


"""
Parameters for the variational multivariate gaussian distribution
"""
@def gaussianparametersfields begin
    μ::Vector{Float64} # Mean for variational distribution
    η_1::Vector{Float64}#Natural Parameter #1
    Σ::Matrix{Float64} # Covariance matrix of variational distribution
    η_2::Matrix{Float64} #Natural Parameter #2
end
"""
Function for initialisation of the variational multivariate parameters
"""
function initGaussian!(model::GPModel,μ_init::Vector{Float64})
    #Initialize gaussian parameters and check for consistency
    if µ_init == [0.0] || length(µ_init) != model.nFeatures
      if model.verbose > 2
        println("***Initial mean of the variational distribution is sampled from a multivariate normal***")
      end
      model.μ = randn(model.nFeatures)
    else
      model.μ = μ_init
    end
    model.Σ = Matrix{Float64}(I,model.nFeatures,model.nFeatures)
    model.η_2 = -0.5*inv(model.Σ)
    model.η_1 = -2.0*model.η_2*model.μ
end
"""
    Parameters defining the available functions of the model
"""
@def functionfields begin
    #Functions
    train::Function #Model train for a certain number of iterations
    fstar::Function #Return the parameters of the latent variable f for a prediction point x
    predict::Function
    predictproba::Function
    elbo::Function
    Plotting::Function
end
"""
Default function to estimate convergence, based on a window on the variational parameters
"""
function DefaultConvergence(model::GPModel,iter::Integer)
    #Default convergence function
    if iter == 1
        if typeof(model) <: MultiClassGPModel
            model.prev_params = vcat(broadcast((μ,diagΣ)->[μ;diagΣ],model.μ,diag.(model.Σ))...)
        else
            model.prev_params = [model.μ;diag(model.Σ)]
        end
        push!(model.evol_conv,Inf)
        return Inf
    end
    if typeof(model) <: MultiClassGPModel
        new_params = vcat(broadcast((μ,diagΣ)->[μ;diagΣ],model.μ,diag.(model.Σ))...)
    else
        new_params = [model.μ;diag(model.Σ)]
    end
    push!(model.evol_conv,mean(abs.(new_params-model.prev_params)./((abs.(model.prev_params)+abs.(new_params))./2.0)))
    model.prev_params = new_params;
    if model.Stochastic
        return mean(model.evol_conv[max(1,iter-model.SmoothingWindow+1):end])
    else
        return model.evol_conv[end]
    end
end

"""
    Appropriately assign the functions
"""
function initFunctions!(model::GPModel)
    #Initialize all functions according to the type of models
    model.train = function(;iterations::Integer=0,callback=0,convergence=DefaultConvergence)
        train!(model;iterations=iterations,callback=callback,Convergence=convergence)
    end
    model.fstar = function(X_test;covf=true)
        return fstar(model,X_test,covf=covf)
    end
    model.predict = function(X_test)
        if !model.Trained
            error("Model has not been trained! Please run .train() beforehand")
            return
        end
        if model.ModelType == BSVM
            probitpredict(model,X_test)
        elseif model.ModelType == XGPC
            logitpredict(model,X_test)
        elseif model.ModelType == Regression
            regpredict(model,X_test)
        elseif model.ModelType == StudentT
            studenttpredict(model,X_test)
        elseif typeof(model) <: MultiClassGPModel
            multiclasspredict(model,X_test)
        end
    end
    model.predictproba = function(X_test)
        if !model.Trained
            error("Model has not been trained! Please run .train() before hand")
            return
        end
        if model.ModelType == BSVM
            probitpredictproba(model,X_test)
        elseif model.ModelType == XGPC
            logitpredictproba(model,X_test)
        elseif model.ModelType == Regression
            regpredictproba(model,X_test)
        elseif model.ModelType == StudentT
            studenttpredictproba(model,X_test)
        elseif typeof(model) <: MultiClassGPModel
            multiclasspredictproba(model,X_test)
        end
    end
    model.elbo = function()
        return ELBO(model)
    end
    model.Plotting = function(;option::String="All")
        ##### TODO ####
    end
end
"""
    Parameters for the linear GPs
"""
@def linearfields begin
    Intercept::Bool
    invΣ::Matrix{Float64} #Inverse Prior Matrix
end
"""
    Initialization of the linear model
"""
function initLinear!(model::GPModel,Intercept::Bool)
    model.Intercept = Intercept;
    model.nFeatures = size(model.X,2);
    if model.Intercept
      model.Intercept = true;
      model.nFeatures += 1
      model.X = [ones(Float64,model.nSamples) model.X]
    end
end
"""
    Parameters of the variational distribution of the augmented variable
"""
@def latentfields begin
    α::Vector{Float64}
    θ::Vector{Float64}
end

"Initialize the latent variables"
function initLatentVariables!(model::FullBatchModel)
    model.α = abs.(rand(model.nSamples))*2;
    model.θ = zeros(Float64,model.nSamples)
end

"Initialize the latent variables"
function initLatentVariables!(model::SparseModel)
    model.α = abs.(rand(model.nSamplesUsed))*2;
    model.θ = zeros(Float64,model.nSamplesUsed)
end

"""
    Parameters for online setting
"""
@def onlinefields begin
    Sequential::Bool #Defines if we know how many point will be treated at the beginning
    alldataparsed::Bool #Check if all data has been treated
    lastindex::Int64
    kmeansalg::KMeansAlg # Online KMean algorithm
    indpoints_updated::Bool#Trigger for matrix computations
    m::Int64 #Number of wanted inducing points
    Kmm::Matrix{Float64} #Kernel matrix
    invKmm::Matrix{Float64} #Inverse Kernel matrix of inducing points
    Ktilde::Vector{Float64} #Diagonal of the covariance matrix between inducing points and generative points
    κ::Matrix{Float64} #Kmn*invKmm
end

"""
Function for initiating online parameters
"""
function initOnline!(model,alg::KMeansAlg,Sequential::Bool,m::Int64)
    model.m = m
    model.kmeansalg = alg
    model.Sequential = Sequential
    model.alldataparsed = false
    model.lastindex=1
    if Sequential
        if typeof(alg) <: StreamOnline || typeof(alg) <: DataSelection
            # newbatchsize = min(max(15,floor(Int64,(model.m-15)/5.0))-1,model.nSamples-model.lastindex)
            newbatchsize = min(model.nSamplesUsed-1,model.nSamples-model.lastindex)
            model.MBIndices = model.lastindex:(model.lastindex+newbatchsize)
            init!(model.kmeansalg,model.X[model.MBIndices,:],model.y[model.MBIndices],model,model.m)
        else
            @assert model.nSamples >= model.m
            newbatchsize = min(model.m-1,model.nSamples-model.lastindex)
            model.MBIndices = model.lastindex:(model.lastindex+newbatchsize)
            init!(model.kmeansalg,model.X[model.MBIndices,:],model.y[model.MBIndices],model,model.m)
        end
    else
        model.MBIndices = StatsBase.sample(1:model.nSamples,model.m,replace=false) #Sample nSamplesUsed indices for the minibatches
        init!(model.kmeansalg,model.X,model.y,model,model.m)
    end
    model.m = model.kmeansalg.k
    model.nFeatures = model.m
    model.indpoints_updated = true
end

"""
    Parameters for the sampling method
"""
@def samplingfields begin
    burninsamples::Integer
    samplefrequency::Integer
    samplehistory_α::Vector{Vector{Float64}}
    samplehistory_f::Vector{Vector{Float64}}
    estimate::Vector{Vector{Float64}}
    pgsampler::PolyaGammaDist
end

function initSampling!(model,burninsamples,samplefrequency)
    model.burninsamples = burninsamples
    model.samplefrequency = samplefrequency
    model.samplehistory_f = Vector{Vector{Float64}}()
    model.samplehistory_α = Vector{Vector{Float64}}()
    model.estimate = Vector{Vector{Float64}}()
    model.pgsampler = PolyaGammaDist()
end
