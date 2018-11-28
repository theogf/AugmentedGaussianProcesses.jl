
"""
    Parameters for the kernel parameters, including the covariance matrix of the prior
"""
@def multiclasskernelfields begin
    IndependentGPs::Bool
    kernel::Vector{Kernel} #Kernels function used
    Knn::Vector{Symmetric{Float64,Matrix{Float64}}} #Kernel matrix of the GP prior
    invK::Vector{Symmetric{Float64,Matrix{Float64}}} #Inverse Kernel Matrix for the nonlinear case
end
"""
Function initializing the kernelfields
"""
function initMultiClassKernel!(model::GPModel,kernel,IndependentGPs)
    #Initialize parameters common to all models containing kernels and check for consistency
    if kernel == 0
      @warn "No kernel indicated, a rbf kernel function with lengthscale 1 is used"
      kernel = RBFKernel(1.0)
    end
    model.IndependentGPs = IndependentGPs
    if model.IndependentGPs
        model.kernel = [deepcopy(kernel) for i in 1:model.K]
    else
        model.kernel = [deepcopy(kernel)]
    end
    model.nFeatures = model.nSamples
end

"""
Parameters for the multiclass version of the classifier based of softmax
"""
@def multiclassfields begin
    Y::Vector{SparseVector{Int64}} #Mapping from instances to classes
    y_class::Vector{Int64}
    K::Int64 #Number of classes
    KStochastic::Bool #Stochasticity in the number of classes
    nClassesUsed::Int64 #Size of class subset
    KStochCoeff::Float64 #Scaling factor for updates
    KIndices::Vector{Int64} #Indices of class subset
    K_map::Vector{Any} #Mapping from the subset of samples to the subset of classes
    class_mapping::Vector{Any} # Classes labels mapping
    ind_mapping::Dict{Any,Int} # Mapping from label to index
    μ::Vector{Vector{Float64}} #Mean for each class
    η_1::Vector{Vector{Float64}} #Natural parameter #1 for each class
    Σ::Vector{Symmetric{Float64,Matrix{Float64}}} #Covariance matrix for each class
    η_2::Vector{Symmetric{Float64,Matrix{Float64}}} #Natural parameter #2 for each class
    c::Vector{Vector{Float64}} #Sqrt of the expectation of f^2
    α::Vector{Float64} #Gamma shape parameters
    β::Vector{Float64} #Gamma rate parameters
    θ::Vector{Vector{Float64}} #Expectations of PG
    γ::Vector{Vector{Float64}} #Poisson rate parameters
end

"""
    Return a matrix of NxK with one of K vectors
"""
function one_of_K_mapping(y)
    y_values = unique(y)
    Y = [spzeros(length(y)) for i in 1:length(y_values)]
    y_class = zeros(Int64,length(y))
    for i in 1:length(y)
        for j in 1:length(y_values)
            if y[i]==y_values[j]
                Y[j][i] = 1;
                y_class[i] = j;
                break;
            end
        end
    end
    ind_values = Dict(value => key for (key,value) in enumerate(y_values))
    return Y,y_values,ind_values,y_class
end

"""
    Initialise the parameters of the multiclass model
"""
function initMultiClass!(model,Y,y_class,y_mapping,ind_mapping,KStochastic,nClassesUsed)
    model.K = length(y_mapping)
    model.Y = Y
    model.KStochastic = KStochastic
    if KStochastic
        if nClassesUsed >= model.K || nClassesUsed <= 0
            @warn "The number of classes used is greater or equal to the number of classes or less than 0, setting back to the classical class batch method"
            model.KStochastic = false;  model.KStochCoeff = 1.0;
            model.nClassesUsed = model.K; model.KIndices = collect(1:model.K);
        else
            model.nClassesUsed = nClassesUsed
            model.KStochCoeff = model.K/model.nClassesUsed
        end
    else
        model.KIndices = collect(1:model.K); model.KStochCoeff = 1.0
        model.nClassesUsed = model.K
    end
    model.class_mapping = y_mapping
    model.ind_mapping = ind_mapping
    model.y_class = y_class
end


function initMultiClassVariables!(model,μ_init)
    if µ_init == [0.0] || length(µ_init) != model.nFeatures
      model.μ = [zeros(model.nFeatures) for _ in 1:model.K]
    else
      model.μ = [μ_init for _ in 1:model.K]
    end
    model.Σ = [Symmetric(Matrix{Float64}(I,model.nFeatures,model.nFeatures)) for _ in 1:model.K]
    model.η_2 = -inv.(model.Σ)*0.5
    model.η_1 = -2.0*model.η_2.*model.μ
    if model.Stochastic
        model.α = model.K*ones(model.nSamples)
        model.β = model.K*ones(model.nSamplesUsed)
        model.θ = [abs.(rand(model.nSamplesUsed))*2 for i in 1:model.K]
        model.γ = [abs.(rand(model.nSamplesUsed)) for i in 1:model.K]
        model.c = [ones(Float64,model.nSamplesUsed) for i in 1:model.K]
    else
        model.α = model.K*ones(model.nSamples)
        model.β = model.K*ones(model.nSamples)
        model.θ = [abs.(rand(model.nSamples))*2 for i in 1:(model.nClassesUsed)]
        model.γ = [abs.(rand(model.nSamples)) for i in 1:model.nClassesUsed]
        model.c = [ones(Float64,model.nSamples) for i in 1:model.nClassesUsed]
    end
end

"Reinitialize vector size after MCInitialization"
function reinit_variational_parameters!(model)
        model.α = model.K*ones(model.nSamples)
        model.β = model.K*ones(model.nSamplesUsed)
        model.θ = [abs.(rand(model.nSamplesUsed))*2 for i in 1:(model.nClassesUsed)]
        model.γ = [abs.(rand(model.nSamplesUsed)) for i in 1:model.nClassesUsed]
        model.c = [ones(Float64,model.nSamplesUsed) for i in 1:model.nClassesUsed]
        model.Ktilde = [ones(Float64,model.nSamplesUsed) for i in 1:model.nClassesUsed]
        model.κ = [Matrix{Float64}(undef,model.nSamplesUsed,model.m) for i in 1:model.nClassesUsed]
        model.Knm = [Matrix{Float64}(undef,model.nSamplesUsed,model.m) for i in 1:model.nClassesUsed]
end

"""
    Parameters necessary for the sparse inducing points method
"""
@def multiclass_sparsefields begin
    m::Int64 #Number of inducing points
    inducingPoints::Vector{Matrix{Float64}} #Inducing points coordinates for the Big Data GP
    OptimizeInducingPoints::Bool #Flag for optimizing the points during training
    optimizer::Optimizer #Optimizer for the inducing points
    nInnerLoops::Int64 #Number of updates for converging α and γ
    Kmm::Vector{Symmetric{Float64,Matrix{Float64}}} #Kernel matrix
    invKmm::Vector{Symmetric{Float64,Matrix{Float64}}} #Inverse Kernel matrix of inducing points
    Knm::Vector{Matrix{Float64}}
    Ktilde::Vector{Vector{Float64}} #Diagonal of the covariance matrix between inducing points and generative points
    κ::Vector{Matrix{Float64}} #Kmn*invKmm
end
"""
Function initializing the multiclass sparsefields parameters
"""
function initMultiClassSparse!(model::GPModel,m::Int64,optimizeIndPoints::Bool)
    #Initialize parameters for the sparse model and check consistency
    minpoints = 56;
    if m > model.nSamples
        @warn "There are more inducing points than actual points, setting it to 10%"
        m = min(minpoints,model.nSamples÷10)
    elseif m == 0
        @warn "Number of inducing points was not manually set, setting it to 10% of the datasize (minimum of $minpoints points)"
        m = min(minpoints,model.nSamples÷10)
    end
    model.m = m; model.nFeatures = model.m;
    model.OptimizeInducingPoints = optimizeIndPoints
    model.optimizer = Adam();
    model.nInnerLoops = 5;
    Ninst_per_K = countmap(model.y)
    Ninst_per_K = [Ninst_per_K[model.class_mapping[i]] for i in 1:model.K]
    if model.verbose>2
        println("$(now()): Starting determination of inducing points through KMeans algorithm")
    end
    if model.IndependentGPs
        model.inducingPoints = Ind_KMeans.(model.nSamples,Ninst_per_K,model.Y,[model.X],model.m)
    else
        model.inducingPoints = [KMeansInducingPoints(model.X,model.m,nMarkov=10)]
    end
    if model.verbose>2
        println("$(now()): Inducing points determined through KMeans algorithm")
    end
    if model.IndependentGPs
        model.Kmm = [Symmetric(Matrix{Float64}(undef,model.m,model.m)) for i in 1:model.K]
        model.invKmm = [Symmetric(Matrix{Float64}(undef,model.m,model.m)) for i in 1:model.K]
        model.Ktilde = [ones(Float64,model.nSamplesUsed) for i in 1:model.K]
        model.κ = [Matrix{Float64}(undef,model.nSamplesUsed,model.m) for i in 1:model.K]
        model.Knm = [Matrix{Float64}(undef,model.nSamplesUsed,model.m) for i in 1:model.K]
    else
        model.Kmm = [Symmetric(Matrix{Float64}(undef,model.m,model.m))]
        model.invKmm = [Symmetric(Matrix{Float64}(undef,model.m,model.m))]
        model.Ktilde = [ones(Float64,model.nSamplesUsed)]
        model.κ = [Matrix{Float64}(undef,model.nSamplesUsed,model.m)]
        model.Knm = [Matrix{Float64}(undef,model.nSamplesUsed,model.m)]
    end
end

"Function to obtain the weighted KMeans for one class"
function Ind_KMeans(nSamples::Int64,N_inst::Int64,Y::SparseVector{Int64},X,m::Int64)
    K_corr = nSamples/N_inst-1.0
    kweights = [Y...].*(K_corr-1.0).+(1.0)
    return KMeansInducingPoints(X,m,nMarkov=10,kweights=kweights)
end


"""
    Parameters for multiclass stochastic optimization
"""
@def multiclassstochasticfields begin
    nSamplesUsed::Int64 #Size of the minibatch used
    StochCoeff::Float64 #Stochastic Coefficient
    MBIndices::Vector{Int64} #MiniBatch Indices
    #Flag for adaptative learning rate for the SVI
    AdaptiveLearningRate::Bool
      κ_s::Float64 #Parameters for decay of learning rate (iter + κ)^-τ in case adaptative learning rate is not used
      τ_s::Float64
    ρ_s::Vector{Float64} #Learning rate for CAVI
    g::Vector{Vector{Float64}} # g & h are expected gradient value for computing the adaptive learning rate and τ is an intermediate
    h::Vector{Float64}
    τ::Vector{Float64}
    SmoothingWindow::Int64
end
"""
    Function initializing the stochasticfields parameters
"""
function initMultiClassStochastic!(model::GPModel,AdaptiveLearningRate,batchsize,κ_s,τ_s,SmoothingWindow)
    #Initialize parameters specific to models using SVI and check for consistency
    model.Stochastic = true; model.nSamplesUsed = batchsize; model.AdaptiveLearningRate = AdaptiveLearningRate;
    model.nInnerLoops = 10;
    model.κ_s = κ_s; model.τ_s = τ_s; model.SmoothingWindow = SmoothingWindow; model.ρ_s = ones(model.K)
    if (model.nSamplesUsed <= 0 || model.nSamplesUsed > model.nSamples)
        @warn "Invalid value for the batchsize : $batchsize, setting it to the number of inducing points"
        model.nSamplesUsed = model.m;
    end
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    model.τ = 10.0*ones(Float64,model.K);
end
