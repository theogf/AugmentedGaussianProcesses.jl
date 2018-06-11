
"""
    Parameters for multiclass stochastic optimization
"""
@def multiclassstochasticfields begin
    nSamplesUsed::Int64 #Size of the minibatch used
    StochCoeff::Float64 #Stochastic Coefficient
    MBIndices #MiniBatch Indices
    #Flag for adaptative learning rate for the SVI
    AdaptiveLearningRate::Bool
      κ_s::Float64 #Parameters for decay of learning rate (iter + κ)^-τ in case adaptative learning rate is not used
      τ_s::Float64
    ρ_s::Array{Float64,1} #Learning rate for CAVI
    g::Array{Array{Float64,1},1} # g & h are expected gradient value for computing the adaptive learning rate and τ is an intermediate
    h::Array{Float64,1}
    τ::Array{Float64,1}
    SmoothingWindow::Int64
end
"""
    Function initializing the stochasticfields parameters
"""
function initMultiClassStochastic!(model::GPModel,AdaptiveLearningRate,BatchSize,κ_s,τ_s,SmoothingWindow)
    #Initialize parameters specific to models using SVI and check for consistency
    model.Stochastic = true; model.nSamplesUsed = BatchSize; model.AdaptiveLearningRate = AdaptiveLearningRate;
    model.nInnerLoops = 10;
    model.κ_s = κ_s; model.τ_s = τ_s; model.SmoothingWindow = SmoothingWindow;
    if (model.nSamplesUsed <= 0 || model.nSamplesUsed > model.nSamples)
################### TODO MUST DECIDE FOR DEFAULT VALUE OR STOPPING STOCHASTICITY ######
        warn("Invalid value for the BatchSize : $BatchSize, assuming a full batch method")
        model.nSamplesUsed = model.nSamples; model.Stochastic = false;
    end
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    model.τ = 50.0*ones(Float64,model.K);
end

"""
Parameters for the multiclass version of the classifier based of softmax
"""
@def multiclassfields begin
    Y::Array{SparseVector,1} #Mapping from instances to classes
    y_class::Array{Int64,1}
    K::Int64 #Number of classes
    KStochastic::Bool #Stochasticity in the number of classes
    class_mapping::Array{Any,1} # Classes labels mapping
    μ::Array{Array{Float64,1}} #Mean for each class
    η_1::Array{Array{Float64,1}} #Natural parameter #1 for each class
    ζ::Array{Array{Float64,2}} #Covariance matrix for each class
    η_2::Array{Array{Float64,2}} #Natural parameter #2 for each class
    α::Array{Float64,1} #Gamma shape parameters
    β::Array{Float64,1} #Gamma rate parameters
    θ::Array{Array{Float64,1}} #Expectations of PG
    γ::Array{Array{Float64,1}} #Poisson rate parameters
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
    return Y,y_values,y_class
end

"""
    Initialise the parameters of the multiclass model
"""

function initMultiClass!(model,Y,y_class,y_mapping)
    model.K = length(y_mapping)
    model.Y = Y
    model.class_mapping = y_mapping
    model.y_class = y_class
end


function initMultiClassVariables!(model,μ_init)
    if µ_init == [0.0] || length(µ_init) != model.nFeatures
      if model.VerboseLevel > 2
        warn("Initial mean of the variational distribution is sampled from a multivariate normal distribution")
      end
      model.μ = [randn(model.nFeatures) for i in 1:model.K]
    else
      model.μ = [μ_init for i in 1:model.K]
    end
    model.ζ = [eye(model.nFeatures) for i in 1:model.K]
    model.η_2 = broadcast(x->-0.5*inv(x),model.ζ)
    model.η_1 = -2.0*model.η_2.*model.μ
    if model.Stochastic
        model.α = 0.5*ones(model.nSamplesUsed)
        model.β = model.K*ones(model.nSamplesUsed)
        model.θ = [abs.(rand(model.nSamplesUsed))*2 for i in 1:(model.K+1)]
        model.γ = [abs.(rand(model.nSamplesUsed)) for i in 1:model.K]
    else
        model.α = 0.5*ones(model.nSamples)
        model.β = model.K*ones(model.nSamples)
        model.θ = [abs.(rand(model.nSamples))*2 for i in 1:(model.K+1)]
        # model.γ = [zeros(model.nSamples) for i in 1:model.K]
        model.γ = [abs.(rand(model.nSamples)) for i in 1:model.K]
    end
end

"""
    Parameters necessary for the sparse inducing points method
"""
@def multiclass_sparsefields begin
    m::Int64 #Number of inducing points
    inducingPoints::Array{Array{Float64,2},1} #Inducing points coordinates for the Big Data GP
    OptimizeInducingPoints::Bool #Flag for optimizing the points during training
    optimizer::Optimizer #Optimizer for the inducing points
    nInnerLoops::Int64 #Number of updates for converging α and γ
    Kmm::Array{Array{Float64,2},1} #Kernel matrix
    invKmm::Array{Array{Float64,2},1} #Inverse Kernel matrix of inducing points
    Ktilde::Array{Array{Float64,1},1} #Diagonal of the covariance matrix between inducing points and generative points
    κ::Array{Array{Float64,2},1} #Kmn*invKmm
end
"""
Function initializing the multiclass sparsefields parameters
"""
function initMultiClassSparse!(model::GPModel,m,optimizeIndPoints)
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
    model.optimizer = Adam();
    model.nInnerLoops = 1;
    Ninst_per_K = countmap(model.y)
    model.inducingPoints= [zeros(model.m,size(model.X,2)) for i in 1:model.K]
    if model.VerboseLevel>2
        println("$(now()): Starting determination of inducing points through KMeans algorithm")
    end
    # for k in 1:model.K
    #     K_corr = model.nSamples/Ninst_per_K[model.class_mapping[k]]-1.0
    #     weights = [model.Y[k]...].*(K_corr-1.0).+(1.0)
    #     model.inducingPoints[k] = KMeansInducingPoints(model.X,model.m,10,weights=weights)
    # end
    indpoints = KMeansInducingPoints(model.X,model.m,10)
    model.inducingPoints = [indpoints for i in 1:model.K]
    if model.VerboseLevel>2
        println("$(now()): Inducing points determined through KMeans algorithm")
    end
end
