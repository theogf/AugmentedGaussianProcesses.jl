include("logistic.jl")
include("bayesiansvm.jl")

const ClassificationLikelihood = BernoulliLikelihood

function (l::BernoulliLikelihood)(y::Real, f::Real)
    return pdf(l(f), y)
end

function init_local_vars(::BernoulliLikelihood, batchsize::Int, T::DataType=Float64)
    return (; c=rand(T, batchsize), θ=zeros(T, batchsize))
end

function compute_proba(
    l::BernoulliLikelihood{L}, μ::AbstractVector{T}, σ²::AbstractVector{T}
) where {L,T<:Real}
    N = length(μ)
    pred = zeros(T, N)
    σ²_pred = zeros(T, N)
    for i in 1:N
        x = pred_nodes .* sqrt(max(σ²[i], zero(T))) .+ μ[i]
        pred[i] = dot(pred_weights, l.invlink.(x))
        σ²_pred[i] = max(dot(pred_weights, abs2.(l.invlink.(x))) - pred[i]^2, zero(T))
    end
    return pred, σ²_pred
end

# Return the labels in a vector of vectors for multiple outputs
function treat_labels!(y::AbstractVector{<:Real}, ::BernoulliLikelihood)
    labels = unique(y)
    # y isa AbstractVector{<:Union{Int,Bool}} || error("y labels should be Int")
    if sort(Int64.(labels)) == [0; 1]
        return sign.(y .- 0.5)
    elseif sort(Int64.(labels)) == [-1; 1]
        return y
    else
        throw(ArgumentError("Labels of y should be binary {-1,1} or {0,1}"))
    end
end

function treat_labels!(::AbstractVector, ::BernoulliLikelihood)
    return error(
        "For classification target(s) should be real valued (Bool, Integer or Float)"
    )
end

predict_y(::BernoulliLikelihood, μ::AbstractVector{<:Real}) = μ .> 0
predict_y(::BernoulliLikelihood, μ::Tuple{<:AbstractVector}) = only(μ) .> 0
