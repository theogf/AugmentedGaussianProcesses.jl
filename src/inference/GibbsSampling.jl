mutable struct GibbsSampling{T<:Real} <: Inference{T}

end





function post_process!(model::GP{<:Likelihood,GibbsSampling})
    model.μ = squeeze(mean(hcat(model.inference.samples...),2),2)
    model.Σ = cov(hcat(model.samples...),2)
end
