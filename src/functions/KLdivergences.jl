""" Compute KL divergence for Polya-Gamma variables """
function PolyaGammaKL(model::GP)
    return model.inference.ρ*sum(broadcast((c,θ)->-0.5*c.^2 .* θ .+ log.(cosh.(0.5.*c)),model.c,model.θ))
end
