abstract type MultiClassLikelihood{T<:Real} <: Likelihood{T} end

## Return the labels in a vector of vectors for multiple outputs ##
function treat_labels!(y::AbstractArray{T,N},likelihood::L) where {T,N,L<:MultiClassLikelihood}
    @assert N <= 1 "Target should be a vector of labels"
    likelihood = init_multiclass_likelihood(likelihood,y)
    return likelihood.Y,length(likelihood.Y),likelihood
end

function init_multiclass_likelihood(likelihood::L,y::AbstractVector) where {L<:MultiClassLikelihood}
    L(one_of_K_mapping(y)...)
end

view_y(l::MultiClassLikelihood,y::AbstractVector,i::AbstractVector) = view(view.(y,[i]),:)


""" Given the labels, return one hot encoding, and the mapping of each class """
function one_of_K_mapping(y)
    y_values = unique(y)
    Y = [falses(length(y)) for i in 1:length(y_values)]
    y_class = zeros(Int64,length(y))
    for i in 1:length(y)
        for j in 1:length(y_values)
            if y[i]==y_values[j]
                Y[j][i] = true;
                y_class[i] = j;
                break;
            end
        end
    end
    ind_values = Dict(value => key for (key,value) in enumerate(y_values))
    return Y,y_values,ind_values,y_class
end

function compute_proba(l::MultiClassLikelihood{T},μ::AbstractVector{<:AbstractVector{T}},σ²::AbstractVector{<:AbstractVector{T}},nSamples::Integer=200) where {T<:Real}
    K = length(μ) # Number of classes
    n = length(μ[1]) # Number of test points
    μ = hcat(μ...) # Concatenate means together
    μ = [μ[i,:] for i in 1:n] # Create one vector per sample
    σ² = hcat(σ²...) # Concatenate variances together
    σ² = [σ²[i,:] for i in 1:n] # Create one vector per sample
    pred = zeros(T,n,K) # Empty container for the predictions
    for i in 1:n
            # p = MvNormal(μ[i],sqrt.(abs.(σ²[i])))
            # p = MvNormal(μ[i],sqrt.(max.(eps(T),σ²[i]))) #WARNING DO NOT USE VARIANCE
            pred[i,:] .= pdf(l,μ[i])
            # for _ in 1:nSamples
            # end
    end
    return DataFrame(pred,Symbol.(l.class_mapping))
end

function expec_logpdf(model::AbstractGP{T,<:MultiClassLikelihood,<:NumericalVI}) where {T}
    compute_log_expectations(model)
end

log_likelihood(l::MultiClassLikelihood,y::AbstractVector,fs) where {T<:Real} = logpdf.(l,y,[getindex.(fs,i) for i in 1:length(y)])

function grad_log_likelihood(l::Likelihood,y::AbstractVector,fs)
    grad_logpdf.(l,y,[getindex.(fs,i) for i in 1:length(y)])
end



include("softmax.jl")
include("logisticsoftmax.jl")
