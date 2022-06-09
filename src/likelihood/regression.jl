abstract type RegressionLikelihood <: AbstractLikelihood end

include("gaussian.jl")
include("studentt.jl")
include("laplace.jl")
include("heteroscedastic.jl")
include("matern.jl")

### Return the labels in a vector of vectors for multiple outputs
function treat_labels!(
    y::AbstractVector{T}, ::Union{RegressionLikelihood,HeteroscedasticGaussianLikelihood}
) where {T}
    T <: Real || throw(ArgumentError("For regression target(s) should be real valued"))
    return y
end

predict_y(::RegressionLikelihood, μ::AbstractVector{<:Real}) = μ
predict_y(::RegressionLikelihood, μ::Tuple{<:AbstractVector}) = only(μ)
