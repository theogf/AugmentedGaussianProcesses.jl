""" Class for sparse variational Gaussian Processes """
mutable struct OnlineSVGP{
    T<:Real,
    TLikelihood<:Likelihood{T},
    TInference<:Inference{T},
    TData<:AbstractDataContainer,
    N,
} <: AbstractGP{T,TLikelihood,TInference,N}
    data::TData
    f::NTuple{N,OnlineVarLatent{T}}
    likelihood::TLikelihood
    inference::TInference
    verbose::Int64
    atfrequency::Int64
    trained::Bool
end

"""
    OnlineSVGP(kernel, likelihood, inference, Zalg)
Create a Online Sparse Variational Gaussian Process model
Argument list :

**Mandatory arguments**
 - `kernel` : covariance function, can be either a single kernel or a collection of kernels for multiclass and multi-outputs models
 - `likelihood` : likelihood of the model, currently implemented : Gaussian, Bernoulli (with logistic link), Multiclass (softmax or logistic-softmax) see [`Likelihood`](@ref)
 - `inference` : inference for the model, can be analytic, numerical or by sampling, check the model documentation to know what is available for your likelihood see [`Inference`](@ref)
 - `ZAlg` : Algorithm to add automatically inducing points, `CircleKMeans` by default, options are : `OfflineKMeans`, `StreamingKMeans`, `Webscale`
 - `nLatent` : Number of needed latent `f`
**Optional arguments**
 - `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
 - `Autotuning` : Flag for optimizing hyperparameters
 - `atfrequency` : Choose how many variational parameters iterations are between hyperparameters optimization
 - `optimiser` : Flux optimizer
"""
function OnlineSVGP(
    kernel::Kernel,
    likelihood::Likelihood,
    inference::Inference,
    Z::AbstractInducingPoints = CircleKMeans(0.9, 0.8);
    verbose::Integer = 0,
    optimiser = ADAM(0.01),
    atfrequency::Integer = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
    IndependentPriors::Bool = true,
    Zoptimiser = nothing,
    T::DataType = Float64,
)
    data = OnlineDataContainer()
    inference isa AnalyticVI || error("The inference object should be of type `AnalyticVI`")
    implemented(likelihood, inference) || error("The $likelihood is not compatible or implemented with the $inference")

    if isa(optimiser, Bool)
        optimiser = optimiser ? ADAM(0.01) : nothing
    end

    Z = Z isa OptimIP ? Z : init_Z(Z, Zoptimiser)

    if typeof(mean) <: Real
        mean = ConstantMean(mean)
    elseif typeof(mean) <: AbstractVector{<:Real}
        mean = EmpiricalMean(mean)
    end

    nLatent = num_latent(likelihood)
    latentf = ntuple(_ -> OnlineVarLatent(T, 0, 0, Z, kernel, mean, optimiser), nLatent)
    inference = tuple_inference(inference, nLatent, 0, 0, 0, [], [])
    inference.nIter = 1
    return OnlineSVGP{T,typeof(likelihood),typeof(inference),typeof(data),nLatent}(
        data,
        latentf,
        likelihood,
        inference,
        verbose,
        atfrequency,
        false,
    )
end

function Base.show(
    io::IO,
    model::OnlineSVGP{T,<:Likelihood,<:Inference},
) where {T}
    print(
        io,
        "Online Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ",
    )
end

@traitimpl IsSparse{OnlineSVGP}

Zviews(m::OnlineSVGP) = Zview.(m.f)
objective(m::OnlineSVGP) = ELBO(m)
MBIndices(m::OnlineSVGP) = 1:nSamples(m)
xview(m::OnlineSVGP) = input(m)
yview(m::OnlineSVGP) = output(m)
nFeatures(m::OnlineSVGP) = collect(dim.(getf(m)))



## Accessors to InducingPoints methods
InducingPoints.init(Z::OptimIP, m::OnlineSVGP, gp::OnlineVarLatent) = InducingPoints.init(Z.Z, m, gp)
InducingPoints.init(Z::OIPS, m, gp) = InducingPoints.init(Z, input(m), kernel(gp))

InducingPoints.add_point!(Z::OptimIP, m::OnlineSVGP, gp::OnlineVarLatent) = InducingPoints.add_point!(Z.Z, m, gp)
InducingPoints.add_point!(Z::OIPS, m::OnlineSVGP, gp::OnlineVarLatent) = InducingPoints.add_point!(Z, input(m), kernel(gp))

InducingPoints.remove_point!(Z::OptimIP, m::OnlineSVGP, gp::OnlineVarLatent) = InducingPoints.remove_point!(Z.Z, m, gp)
InducingPoints.remove_point!(Z::OIPS, m::OnlineSVGP, gp::OnlineVarLatent) = InducingPoints.remove_point!(Z, pr_cov(gp), kernel(gp))

opt(Z::OptimIP) = Z.opt
opt(Z::AbstractInducingPoints) = nothing

function update!(opt, Z::AbstractInducingPoints, Z_grads)
    for (z, zgrad) in zip(Z, Z_grads)
        z .+= apply(opt, z, zgrad)
    end
end
