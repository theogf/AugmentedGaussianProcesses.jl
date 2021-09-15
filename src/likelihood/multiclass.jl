abstract type MultiClassLikelihood{T<:Real} <: AbstractLikelihood{T} end

n_latent(l::MultiClassLikelihood) = n_class(l)

n_class(l::MultiClassLikelihood) = l.n_class

## Return the labels in a vector of vectors for multiple outputs ##
function treat_labels!(
    y::AbstractArray{T,N}, likelihood::L
) where {T,N,L<:MultiClassLikelihood}
    N <= 1 "Target should be a vector of labels"
    init_multiclass_likelihood!(likelihood, y)
    return likelihood.Y, num_class(likelihood), likelihood
end

function init_multiclass_likelihood!(l::MultiClassLikelihood, y::AbstractVector)
    if !isdefined(l, :ind_mapping)
        create_mapping!(l, y)
    end
    return create_one_hot!(l, y)
end

view_y(::MultiClassLikelihood, y::AbstractVector, i::AbstractVector) = view.(y, Ref(i))

onehot_to_ind(y::AbstractVector) = findfirst(y .== 1)

function Distributions.loglikelihood(
    l::MultiClassLikelihood, y::AbstractVector, f::AbstractVector
)
    return loglikelihood(l, onehot_to_ind(y), f)
end

function create_mapping!(l::MultiClassLikelihood, y::AbstractVector)
    num_latent = n_latent(l)
    if !isdefined(l, :class_mapping)
        l.class_mapping = unique(y)
        if length(l.class_mapping) <= num_latent &&
           issubset(l.class_mapping, collect(1:num_latent))
            l.class_mapping = collect(1:num_latent)
        elseif length(l.class_mapping) > nClasses
            throw(
                ErrorException(
                    "The number of unique labels in the data : $(l.class_mapping) is not of the same size then the predefined class number ; $num_latent",
                ),
            )
        end
    end
    return l.ind_mapping = Dict(value => key for (key, value) in enumerate(l.class_mapping))
end

# Given the labels, return one hot encoding, and the mapping of each class
function create_one_hot(l::MultiClassLikelihood, y)
    issubset(unique(y), l.class_mapping) || error("Some labels of y are not part of the expect labels")
    Y = falses(length(y), n_class(l))
    for i in 1:length(y)
        for j in 1:n_class(l)
            if y[i] == l.class_mapping[j]
                Y[i, j] = true
                break # Once we found the right one, we get out
            end
        end
    end
    return Y
end

function compute_proba(
    l::MultiClassLikelihood{T},
    μ::AbstractVector{<:AbstractVector{T}},
    σ²::AbstractVector{<:AbstractVector{T}},
    nSamples::Integer=200,
) where {T<:Real}
    K = length(μ) # Number of classes
    n = length(μ[1]) # Number of test points
    μ = hcat(μ...) # Concatenate means together
    μ = [μ[i, :] for i in 1:n] # Create one vector per sample
    σ² = hcat(σ²...) # Concatenate variances together
    σ² = [σ²[i, :] for i in 1:n] # Create one vector per sample
    pred = zeros(T, n, K) # Empty container for the predictions
    for i in 1:n
        # p = MvNormal(μ[i],sqrt.(abs.(σ²[i])))
        # p = MvNormal(μ[i],sqrt.(max.(eps(T),σ²[i]))) #WARNING DO NOT USE VARIANCE
        pred[i, :] .= l(μ[i])
        # for _ in 1:nSamples
        # end
    end
    return NamedTuple{Tuple(Symbol.(l.class_mapping))}(eachcol(pred))
end

function expec_loglike(
    model::AbstractGPModel{T,<:MultiClassLikelihood,<:NumericalVI}
) where {T}
    return compute_log_expectations(model)
end

function Distributions.loglikelihood(l::MultiClassLikelihood, y::AbstractVector, fs)
    return loglikelihood.(l, y, [getindex.(fs, i) for i in 1:length(y)])
end

function ∇loglikehood(l::MultiClassLikelihood, y::AbstractVector, fs)
    return ∇loglikelihood.(l, y, [getindex.(fs, i) for i in 1:length(y)])
end

include("softmax.jl")
include("logisticsoftmax.jl")
