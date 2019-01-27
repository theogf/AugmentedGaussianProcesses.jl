mutable struct GibbsSampling <: Inference

end





function post_process!(model::GP{L,GibbsSampling})
    model.μ = squeeze(mean(hcat(model.inference.samples...),2),2)
    model.Σ = cov(hcat(model.samples...),2)
end
