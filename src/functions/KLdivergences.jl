"""Compute the KL Divergence between the GP Prior and the variational distribution"""
GaussianKL(model::AbstractGP) = sum(broadcast(GaussianKL,model.f,get_Z(model)))

GaussianKL(gp::Abstract_GP,X::AbstractMatrix) = GaussianKL(gp.μ,gp.μ₀(X),gp.Σ,gp.K)

function GaussianKL(μ::AbstractVector{T},μ₀::AbstractVector,Σ::Matrix{T},K::PDMat{T,Matrix{T}}) where {T<:Real}
    0.5*(-logdet(Σ)+logdet(K)+tr(K\Σ)+invquad(K,μ-μ₀)-length(μ))
end

function extraKL(model::VGP)
    return 0
end

function extraKL(model::SVGP)
    return 0
end

"""Return the extra KL term containing the divergence with the GP at time t and t+1"""
function extraKL(model::OnlineSVGP{T}) where {T}
    jitt = T(Jittering())
    L = 0.0
    for gp in model.f
        Kₐₐ = first(gp.σ_k)*(kernelmatrix(gp.kernel,gp.Zₐ,obsdim=1)+jitt*I)
        κₐμ = gp.κₐ*gp.μ
        L += - gp.𝓛ₐ - opt_trace(gp.invDₐ,Kₐₐ) -
        opt_trace(gp.invDₐ,gp.κₐ*(gp.Σ*gp.κₐ'-gp.Kab')) +
        2*dot(gp.η₁,κₐμ) - dot(κₐμ,gp.invDₐ*κₐμ)
    end
     #Precompute this part for the next ELBO
    return 0.5*L
end



InverseGammaKL(α,β,αₚ,βₚ) = GammaKL(α,β,αₚ,βₚ)
"""
    KL(q(ω)||p(ω)), where q(ω) = Ga(α,β) and p(ω) = Ga(αₚ,βₚ)
"""
function GammaKL(α,β,αₚ,βₚ)
    sum((α-αₚ).*digamma(α) .- log.(gamma.(α)).+log.(gamma.(αₚ)) .+  αₚ.*(log.(β).-log.(βₚ)).+α.*(βₚ.-β)./β)
end

"""
    KL(q(ω)||p(ω)), where q(ω) = Po(ω|λ) and p(ω) = Po(ω|λ₀)
"""
function PoissonKL(λ::AbstractVector{T},λ₀::Real) where {T}
    λ₀*length(λ)-(one(T)+log(λ₀))*sum(λ)+sum(xlogx,λ)
end

"""
    KL(q(ω)||p(ω)), where q(ω) = Po(ω|λ) and p(ω) = Po(ω|λ₀) with ψ = E[log(λ₀)]
"""
function PoissonKL(λ::AbstractVector{<:Real},λ₀::AbstractVector{<:Real},ψ::AbstractVector{<:Real})
    sum(λ₀)-sum(λ)+sum(xlogx,λ)-dot(λ,ψ)
end


"""KL(q(ω)||p(ω)), where q(ω) = PG(b,c) and p(ω) = PG(b,0). θ = 𝑬[ω]"""
function PolyaGammaKL(b,c,θ)
    dot(b,logcosh.(0.5*c))-0.5*dot(abs2.(c),θ)
end


"""
    Entropy of GIG variables with parameters a,b and p and omitting the derivative d/dpK_p cf <https://en.wikipedia.org/wiki/Generalized_inverse_Gaussian_distribution#Entropy>
"""
function GIGEntropy(a,b,p)
    sqrtab = sqrt.(a.*b)
    return sum(0.5*log.(a./b))+sum(log.(2*besselk.(p,sqrtab)))+ sum(0.5*sqrtab./besselk.(p,sqrtab).*(besselk.(p+1,sqrtab)+besselk.(p-1,sqrtab)))
end
