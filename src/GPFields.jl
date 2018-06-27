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
    X #Feature vectors
    y #Output (-1,1 for classification, real for regression, matrix for multiclass)
    ModelType::GPModelType; #Type of model
    Name::String #Name of the model
    nSamples::Int64 # Number of data points
    nDim::Int64
    nFeatures::Int64 # Number of features
    noise::Float64  #Regularization parameter of the noise
    ϵ::Float64  #Desired Precision on ||ELBO(t+1)-ELBO(t)||))
    evol_conv::Array{Float64,1} #Used for convergence estimation
    prev_params::Any
    nEpochs::Int64; #Maximum number of iterations
    VerboseLevel::Int64 #Level of printing information
    Stochastic::Bool #Is the model stochastic    #Autotuning parameters
    Autotuning::Bool #Chose Autotuning type
    MaxGradient::Float64 #Maximum value for the gradient clipping
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
function initCommon!(model::GPModel,X,y,noise,ϵ,nEpochs,VerboseLevel,Autotuning,AutotuningFrequency,optimizer)
    @assert (size(y,1)==size(X,1)) "There is a dimension problem with the data size(y)!=size(X)";
    model.X = X; model.y = y;
    @assert noise > 0 "noise should be a positive float"; model.noise = noise;
    @assert ϵ > 0 "ϵ should be a positive float"; model.ϵ = ϵ;
    @assert nEpochs > 0 "nEpochs should be positive"; model.nEpochs = nEpochs;
    @assert (VerboseLevel > -1 && VerboseLevel < 4) "VerboseLevel should be in {0,1,2,3}, here value is $VerboseLevel"; model.VerboseLevel = VerboseLevel;
    model.Autotuning = Autotuning; model.AutotuningFrequency = AutotuningFrequency;
    # model.opt_type = optimizer;
    model.nSamples = size(X,1); #model.nSamplesUsed = model.nSamples;
    model.nDim= size(X,2);
    model.Trained = false; model.Stochastic = false;
    model.TopMatrixForPrediction = 0; model.DownMatrixForPrediction = 0; model.MatricesPrecomputed=false;
    model.MaxGradient = 50;
    model.HyperParametersUpdated = true;
    model.evol_conv = Array{Float64,1}()
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
    g::Array{Float64,1} # g & h are expected gradient value for computing the adaptive learning rate and τ is an intermediate
    h::Float64
    τ::Float64
    SmoothingWindow::Int64
end
"""
    Function initializing the stochasticfields parameters
"""
function initStochastic!(model::GPModel,AdaptiveLearningRate,BatchSize,κ_s,τ_s,SmoothingWindow)
    #Initialize parameters specific to models using SVI and check for consistency
    model.Stochastic = true; model.nSamplesUsed = BatchSize; model.AdaptiveLearningRate = AdaptiveLearningRate;
    model.κ_s = κ_s; model.τ_s = τ_s; model.SmoothingWindow = SmoothingWindow;
    if (model.nSamplesUsed <= 0 || model.nSamplesUsed > model.nSamples)
################### TODO MUST DECIDE FOR DEFAULT VALUE OR STOPPING STOCHASTICITY ######
        warn("Invalid value for the BatchSize : $BatchSize, assuming a full batch method")
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
    invK::Array{Float64,2} #Inverse Kernel Matrix for the nonlinear case
end
"""
Function initializing the kernelfields
"""
function initKernel!(model::GPModel,kernel)
    #Initialize parameters common to all models containing kernels and check for consistency
    if kernel == 0
      warn("No kernel indicated, a rbf kernel function with lengthscale 1 is used")
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
    inducingPoints::Array{Float64,2} #Inducing points coordinates for the Big Data GP
    OptimizeInducingPoints::Bool #Flag for optimizing the points during training
    optimizer::Optimizer #Optimizer for the inducing points
    Kmm::Array{Float64,2} #Kernel matrix
    invKmm::Array{Float64,2} #Inverse Kernel matrix of inducing points
    Ktilde::Array{Float64,1} #Diagonal of the covariance matrix between inducing points and generative points
    κ::Array{Float64,2} #Kmn*invKmm
end
"""
Function initializing the sparsefields parameters
"""
function initSparse!(model::GPModel,m,optimizeIndPoints)
    #Initialize parameters for the sparse model and check consistency
    minpoints = 56;
    if m > model.nSamples
        warn("There are more inducing points than actual points, setting it to 10%")
        m = min(minpoints,model.nSamples÷10)
    elseif m == 0
        warn("Number of inducing points was not manually set, setting it to 10% of the datasize (minimum of $minpoints points)")
        m = min(minpoints,model.nSamples÷10)
    end
    model.m = m; model.nFeatures = model.m;
    model.OptimizeInducingPoints = optimizeIndPoints
    model.optimizer = Adam(α=0.1);
    model.inducingPoints = KMeansInducingPoints(model.X,model.m,10)
    if model.VerboseLevel>1
        println("Inducing points determined through KMeans algorithm")
    end
    model.inducingPoints += rand(Normal(0,0.1),size(model.inducingPoints)...)
end


"""
Parameters for the variational multivariate gaussian distribution
"""
@def gaussianparametersfields begin
    μ::Array{Float64,1} # Mean for variational distribution
    η_1::Array{Float64,1} #Natural Parameter #1
    ζ::Array{Float64,2} # Covariance matrix of variational distribution
    η_2::Array{Float64,2} #Natural Parameter #2
end
"""
Function for initialisation of the variational multivariate parameters
"""
function initGaussian!(model::GPModel,μ_init)
    #Initialize gaussian parameters and check for consistency
    if µ_init == [0.0] || length(µ_init) != model.nFeatures
      if model.VerboseLevel > 2
        warn("Initial mean of the variational distribution is sampled from a multivariate normal distribution")
      end
      model.μ = randn(model.nFeatures)
    else
      model.μ = μ_init
    end
    model.ζ = eye(model.nFeatures)
    model.η_2 = -0.5*inv(model.ζ)
    model.η_1 = -2.0*model.η_2*model.μ
end
"""
    Parameters defining the available functions of the model
"""
@def functionfields begin
    #Functions
    train::Function #Model train for a certain number of iterations
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
            model.prev_params = vcat(broadcast((μ,diagζ)->[μ;diagζ],model.μ,diag.(model.ζ))...)
        else
            model.prev_params = [model.μ;diag(model.ζ)]
        end
        push!(model.evol_conv,Inf)
        return Inf
    end
    if typeof(model) <: MultiClassGPModel
        new_params = vcat(broadcast((μ,diagζ)->[μ;diagζ],model.μ,diag.(model.ζ))...)
    else
        new_params = [model.μ;diag(model.ζ)]
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
    invΣ::Array{Float64,2} #Inverse Prior Matrix
end
"""
    Initialization of the linear model
"""
function initLinear!(model::GPModel,Intercept)
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
    α::Array{Float64,1}
end

function initLatentVariables!(model)
    #Initialize the latent variables
    model.α = abs.(rand(model.nSamples))*2;
end

"""
    Parameters of the variational distribution of the augmented variable
"""
@def EPlatentfields begin
    ν::Array{Float64,1}
    mu::Array{Float64,1}
    α::Array{Float64,1}
    a::Array{Float64,1}
    b::Array{Float64,1}
    θ::Array{Float64,1}
    s::Array{Float64,1}
    Z::Array{Float64,1}
end

function initEPLatentVariables!(model)
    #Initialize the latent variables
    model.ν = zeros(model.nSamples);
    model.mu = zeros(model.nSamples);
    model.α = zeros(model.nSamples);
    model.a = zeros(model.nSamples);
    model.b = zeros(model.nSamples);
    model.θ = zeros(model.nSamples);
    model.s = zeros(model.nSamples);
    model.Z = zeros(model.nSamples);
end
"""
    Parameters for the sampling method
"""
@def samplingfields begin
    burninsamples::Integer
    samplefrequency::Integer
    samplehistory::MVHistory
    estimate::Array{Array{Float64,1},1}
    pgsampler::PolyaGammaDist
end

function initSampling!(model,burninsamples,samplefrequency)
    model.burninsamples = burninsamples
    model.samplefrequency = samplefrequency
    model.samplehistory = MVHistory()
    model.estimate = Array{Array{Float64,1},1}()
    model.pgsampler = PolyaGammaDist()
end
