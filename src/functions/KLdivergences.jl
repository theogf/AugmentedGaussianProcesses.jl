""" Compute KL divergence for Polya-Gamma variables """
function PolyaGammaKL(model::GP)
    return model.inference.ρ*sum(broadcast((c,θ)->-0.5*c.^2 .* θ .+ log.(cosh.(0.5.*c)),model.c,model.θ))
end

function GammaImproperKL(model::GP)
    return model.inference.ρ*sum(-model.likelihood.α.+log(model.likelihood.β[1]).-log.(gamma.(model.likelihood.α)).-(1.0.-model.likelihood.α).*digamma.(model.likelihood.α))
end

function PoissonKL(model::GP)
    return model.inference.ρ*sum(γ->sum(γ.*(log.(γ).-1.0.-digamma.(model.likelihood.α).+log.(model.likelihood.β))+model.likelihood.α./model.likelihood.β),model.likelihood.γ)
end

function PolyaGammaKL(model::VGP{<:AugmentedLogisticSoftMaxLikelihood})
    return sum(broadcast((y,γ,c,θ)->sum((y+γ).*log.(cosh.(0.5.*c))-0.5*(c.^2).*θ),model.likelihood.Y,model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
end


function PolyaGammaKL(model::SVGP{<:AugmentedLogisticSoftMaxLikelihood})
    return sum(broadcast((y,γ,c,θ)->model.inference.ρ*sum((y[model.inference.MBIndices]+γ).*log.(cosh.(0.5.*c))-0.5*(c.^2).*θ),model.likelihood.Y,model.likelihood.γ,model.likelihood.c,model.likelihood.θ))
end
