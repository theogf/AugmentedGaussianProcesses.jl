


function ELBO(model::GPRegression)
    return -0.5*dot(model.y,model.invK*model.y)+0.5logdet(model.invK)-model.nSamples*log(2*pi)
end

function ELBO(model::SparseGPRegression)
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    ELBO = -0.5*model.nSamples*(log(model.γ)+log(2*pi))
    ELBO += -model.StochCoeff*((model.y - model.κ*model.μ)./model.γ).^2
    ELBO += -0.5*model.StochCoeff*model.Ktilde./model.γ
    ELBO += -0.5*model.StochCoeff/model.γ*sum(model.kappa.*model.kappa)
    ELBO += 0.5*(logdet(model.ζ)+logdet(model.invKmm))
    ELBO += -0.5*(sum(model.invKmm.*transpose(model.ζ+model.μ*transpose(model.μ))))
    return -ELBO
end
