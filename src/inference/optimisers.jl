mutable struct RobbinsMonro
    κ::Float64
    τ::Float64
    state::IdDict
end

function RobbinsMonro(κ::Real=0.51, τ::Real=1)
    @assert 0.5 < κ <= 1 "κ should be in the interval (0.5,1]"
    @assert τ > 0 "τ should be positive"
    return RobbinsMonro(κ, τ, IdDict())
end

function Optimise.apply!(o::RobbinsMonro, x, Δ)
    κ = o.κ
    τ = o.τ
    n = get!(o.state, x, 1)
    Δ .*= 1 / (τ + n)^κ
    o.state[x] = n + 1
    return Δ
end

mutable struct ALRSVI
    τ::Int64
    state::IdDict
end

struct ALRSVIBase{T<:Real}
    g::Array{T}
    h::Base.RefValue{T}
    ρ::Base.RefValue{T}
    τ::Base.RefValue{T}
    t::Base.RefValue{Int64}
end

""" Construct Adaptive Learning Rate for Stochastic Variational Inference"""
function ALRSVI(ρ::Real=0.1, τ::Int=100)
    return ALRSVI(ρ, τ, IdDict())
end

function init!(model::AbstractGP{T,L,<:AnalyticVI}) where {T,L}
    for n_s in 1:(model.vi_opt[1].opt.τ)
        model.inference.MBIndices .= StatsBase.sample(
            1:(model.inference.nSamples), model.inference.nMinibatch; replace=false
        )
        model.inference.xview = view(model.X, model.inference.MBIndices, :)
        model.inference.yview = view_y(model.likelihood, model.y, model.inference.MBIndices)
        computeMatrices!(model)
        local_updates!(likelihood(model), yview(model), mean_f(model), var_f(model))
        natural_gradient!.(
            ∇E_μ(model.likelihood, model.inference.vi_opt[1], yview(model)),
            ∇E_Σ(model.likelihood, model.inference.vi_opt[1], yview(model)),
            model.inference,
            model.inference.vi_opt,
            Zviews(model),
            model.f,
        )
        init_ALRSVI!.(model.inference.vi_opt, model.f, τ)
    end
    return finalize_init_ALRSVI!.(model.inference.vi_opt, model.f)
end

function init_ALRSVI!(vi_opt::AVIOptimizer, gp::AbstractLatent{T}, τ) where {T}
    objη₁ = get!(
        vi_opt.opt.state,
        gp.η₁,
        ALRSVIBase(zeros(T, gp.dims), Ref(zero(T)), Ref(1.0), Ref(τ), Ref(1)),
    )
    objη₁.g .+= inf.∇η₁
    objη₁.h[] += norm(inf.∇η₁)

    objη₂ = get!(
        vi_opt.opt.state,
        gp.η₂,
        ALRSVIBase(zeros(T, gp.dims, gp.dims), Ref(zero(T)), Ref(1.0), Ref(τ), Ref(1)),
    )
    objη₂.g .+= inf.∇η₂
    return objη₂.h[] += sum(abs2, inf.∇η₂)
end

function finalize_init_ALRSVI!(vi_opt::AVIOptimizer, gp::AbstractLatent{T}) where {T}
    objη₁ = get(vi_opt.opt.state, gp.η₁)
    objη₁.g ./= obj.τ
    objη₁.h[] /= obj.τ
    objη₁.ρ[] = dot(objη₁.g, objη₁.g) / objη₁.h
    objη₂ = get(vi_opt.opt.state, gp.η₂)
    objη₂.g ./= obj.τ
    objη₂.h[] /= obj.τ
    return objη₂.ρ[] = sum(abs2, objη₂.g) / objη₂.h
end

function Optimise.apply!(opt::ALRSVI, x, Δ)
    # update timestep
    obj = get(opt.state, x)
    obj.t += 1
    obj.g .= (1.0 - 1.0 / obj.τ) * obj.g .+ 1.0 / obj.τ * Δ
    obj.h[] = (1.0 - 1.0 / obj.τ) * obj.h[] + 1.0 / obj.τ * sum(abs2, Δ)
    obj.ρ[] = dot(obj.g, obj.g) / obj.h[]
    obj.τ[] = obj.τ[] * (1 - obj.ρ[]) + 1.0
    return obj.ρ * Δ
end
