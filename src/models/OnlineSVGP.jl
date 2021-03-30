mutable struct OnlineSVGP{
    T<:Real,
    TLikelihood<:AbstractLikelihood,
    TInference<:AbstractInference,
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
    OnlineSVGP(args...; kwargs...)

Online Sparse Variational Gaussian Process

## Arguments 
- `kernel::Kernel` : Covariance function, can be any kernel from KernelFunctions.jl
- `likelihood` : Likelihood of the model. For compatibilities, see [`Likelihood Types`](@ref likelihood_user)
- `inference` : Inference for the model, see the [`Compatibility Table`](@ref compat_table))
- `Zalg` : Algorithm selecting how inducing points are selected

## Keywords arguments
- `verbose` : How much does the model print (0:nothing, 1:very basic, 2:medium, 3:everything)
- `optimiser` : Optimiser used for the kernel parameters. Should be an Optimiser object from the [Flux.jl](https://github.com/FluxML/Flux.jl) library, see list here [Optimisers](https://fluxml.ai/Flux.jl/stable/training/optimisers/) and on [this list](https://github.com/theogf/AugmentedGaussianProcesses.jl/tree/master/src/inference/optimisers.jl). Default is `ADAM(0.001)`
- `atfrequency::Int=1` : Choose how many variational parameters iterations are between hyperparameters optimization
- `mean=ZeroMean()` : PriorMean object, check the documentation on it [`MeanPrior`](@ref meanprior)
- `Zoptimiser` : Optimiser for inducing points locations
- `T::DataType=Float64` : Hint for what the type of the data is going to be.
"""
function OnlineSVGP(
    kernel::Kernel,
    likelihood::AbstractLikelihood,
    inference::AbstractInference,
    Z::AbstractInducingPoints = OIPS(0.9);
    verbose::Integer = 0,
    optimiser = ADAM(0.01),
    atfrequency::Integer = 1,
    mean::Union{<:Real,AbstractVector{<:Real},PriorMean} = ZeroMean(),
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
    model::OnlineSVGP,
) where {T}
    print(
        io,
        "Online Variational Gaussian Process with a $(likelihood(model)) infered by $(inference(model)) ",
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