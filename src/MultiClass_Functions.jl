function variablesUpdate_MultiClass!(model::MultiClass,iter)
    expec_f2 = broadcast((var,m)->var.+m.^2,diag.(model.ζ),model.μ)
    model.θ[1] = [0.5./sqrt(expec_f2[model.y_class[i]][i])*tanh(0.5*sqrt(expec_f2[model.y_class[i]][i])) for i in 1:model.nSamples ];
    # println(mean(model.α),broadcast(mean,model.γ))
    model.γ = broadcast((f,μ)->0.5./(cosh.(0.5.*f).*model.β).*exp.(digamma.(model.α).-0.5.*μ),expec_f2,model.μ)
    model.α = [1+sum(broadcast(x->x[i],model.γ)) for i in 1:model.nSamples]
    model.θ[2:end] = broadcast((γ,f)->0.5.*γ./sqrt.(f).*tanh.(0.5.*sqrt.(f)),model.γ,expec_f2)
    (grad_η_1, grad_η_2) = naturalGradientELBO_MultiClass(model.y,diagm(model.θ[1]),diagm.(model.θ[2:end]),model.invK,model.γ)
    model.η_1 = grad_η_1; model.η_2 = grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = broadcast(x->-0.5*inv(x),model.η_2); model.μ = model.ζ.*model.η_1 #Back to the distribution parameters (needed for α updates)
end

function naturalGradientELBO_MultiClass(Y,θ_0,θ,invK,γ)
    grad_1 = broadcast((y,γ)->0.5*(y-γ),Y,γ)
    grad_2 = broadcast((y,θ)->-0.5*(diagm(y)*θ_0+θ+invK),Y,θ)
    return grad_1,grad_2
end
