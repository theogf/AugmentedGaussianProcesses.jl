"""File containing different utility functions"""

""" Jittering object adapting to the type of the GP """
struct Jittering
function Jittering()
    new()
end
end;

Base.convert(::Type{Float64},::Jittering) = Float64(1e-3)
Base.convert(::Type{Float32},::Jittering) = Float32(1e-2)
Base.convert(::Type{Float16},::Jittering) = Float16(1e-1)

""" delta function `(i,j)`, equal `1` if `i == j`, `0` else """
@inline function δ(i::Integer,j::Integer)
    i == j ? 1.0 : 0.0
end

"""Hadamard product between two arrays of same size"""
function hadamard(A::AbstractArray{<:Real},B::AbstractArray{<:Real})
    A.*B
end

""" Add on place the transpose of a matrix """
function add_transpose!(A::AbstractMatrix{<:Real})
    A .+= A'
end

""" Return the trace of A*B' """
function opt_trace(A::AbstractMatrix{<:Real},B::AbstractMatrix{<:Real})
    dot(A,B)
end

""" Return the diagonal of A*B' """
function opt_diag(A::AbstractMatrix{<:Real},B::AbstractMatrix{<:Real})
    vec(sum(A.*B,dims=2))
end

function GammaImproperKL(model::GP)
    return sum(-model.likelihood.α.+log(model.likelihood.β[1]).-log.(gamma.(model.likelihood.α)).-(1.0.-model.likelihood.α).*digamma.(model.likelihood.α))
end

function PoissonKL(model::GP)
    return sum(γ->sum(γ.*(log.(γ).-1.0.-digamma.(model.likelihood.α).+log.(model.likelihood.β))+model.likelihood.α./model.likelihood.β),model.likelihood.γ)
end

function PolyaGammaKL(model::VGP{<:AugmentedLogisticSoftMaxLikelihood})
    return sum(broadcast((y,γ,c,θ)->sum(Array(y+γ).*log.(cosh.(0.5.*c))-0.5*(c.^2).*θ),model.likelihood.Y,model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
end


function PolyaGammaKL(model::SVGP{<:AugmentedLogisticSoftMaxLikelihood})
    return sum(broadcast((y,γ,c,θ)->sum(Array(y[model.inference.MBIndices]+γ).*log.(cosh.(0.5.*c))-0.5*(c.^2).*θ),model.likelihood.Y,model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
end
