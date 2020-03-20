include("vi_optimizers.jl")
include("analytic.jl")
include("analyticVI.jl")
include("numericalVI.jl")
include("sampling.jl")
include("optimisers.jl")

export RobbinsMonro, ALRSVI


function post_process!(model::AbstractGP{T,<:Likelihood,<:Inference}) where {T}
    nothing
end

Base.length(::Inference) = 1

Base.iterate(l::Inference) = (l,nothing)
Base.iterate(l::Inference, ::Any) = nothing

isstochastic(l::Inference) = l.Stochastic

const GibbsorVI = Union{<:GibbsSampling,<:AnalyticVI}

#Conversion from natural to standard distribution parameters
function global_update!(gp::Abstract_GP) where {T,L}
    gp.Σ .= -0.5*inv(gp.η₂)
    gp.μ .= gp.Σ*gp.η₁
end

function global_update!(gp::_OSVGP) where {T,L}
    gp.Σ = -0.5*inv(gp.η₂)
    gp.μ = gp.Σ*gp.η₁
end


## Default function for getting a view on y
@inline view_y(l::Likelihood,y::AbstractVector,i::AbstractVector) = view(y,i)

## Default function for getting gradient ##
function grad_logpdf(l::Likelihood,y::Real,f::Real)
    ForwardDiff.gradient(x->AugmentedGaussianProcesses.logpdf(l,y,x[1]),[f])[1]
end

function grad_logpdf(l::Likelihood,y::Real,f::AbstractVector)
    ForwardDiff.gradient(x->AugmentedGaussianProcesses.logpdf(l,y,x),f)
end

function hessian_logpdf(l::Likelihood,y::Real,f::Real)
    ForwardDiff.hessian(x->AugmentedGaussianProcesses.logpdf(l,y,x[1]),[f])[1]
end

function hessian_logpdf(l::Likelihood,y::Real,f::AbstractVector)
    ForwardDiff.hessian(x->AugmentedGaussianProcesses.logpdf(l,y,x),f)
end
