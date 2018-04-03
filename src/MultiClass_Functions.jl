function variablesUpdate_MultiClass!(model::MultiClass,iter)
    expec_f2 = broadcast((var,m)->var.+m.^2,diag.(model.ζ),model.μ)
    model.θ[1] = [0.5./sqrt(expec_f2[model.y_class[i]][i])*tanh(0.5*sqrt(expec_f2[model.y_class[i]][i])) for i in 1:model.nSamples ];
    # println(mean(model.α),broadcast(mean,model.γ))
    model.γ = broadcast((f,μ)->0.5./(cosh.(0.5.*f).*model.β).*exp.(digamma.(model.α).-0.5.*μ),expec_f2,model.μ)
    model.α = [1+sum(broadcast(x->x[i],model.γ)) for i in 1:model.nSamples]
    model.θ[2:end] = broadcast((γ,f)->0.5.*γ./sqrt.(f).*tanh.(0.5.*sqrt.(f)),model.γ,expec_f2)
    (grad_η_1, grad_η_2) = naturalGradientELBO_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invK,model.γ)
    model.η_1 = grad_η_1; model.η_2 = grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = broadcast(x->-0.5*inv(x),model.η_2); model.μ = model.ζ.*model.η_1 #Back to the distribution parameters (needed for α updates)
end

function variablesUpdate_MultiClass!(model::SparseMultiClass,iter)
    expec_f2 = broadcast((m,var,kappa,ktilde)->ktilde+sum((kappa*var).*kappa,2)+(kappa*m).^2,model.μ,model.ζ,model.κ,model.Ktilde)
    model.θ[1][model.MBIndices] = [0.5./sqrt(expec_f2[model.y_class[i]][i])*tanh(0.5*sqrt(expec_f2[model.y_class[i]][i])) for i in model.MBIndices ];
    # println(mean(model.α),broadcast(mean,model.γ))
    broadcast((gamma,f,kappa,μ)->gamma[model.MBIndices]=0.5./(cosh.(0.5.*f).*model.β[model.MBIndices]).*exp.(digamma.(model.α[model.MBIndices]).-0.5.*kappa*μ),model.γ,expec_f2,model.κ,model.μ)
    model.α[model.MBIndices] = [1+sum(broadcast(x->x[i],model.γ)) for i in model.MBIndices]
    broadcast((theta,γ,f)->theta[model.MBIndices]=0.5.*γ./sqrt.(f).*tanh.(0.5.*sqrt.(f)),model.θ[2:end],model.γ,expec_f2)
    (grad_η_1, grad_η_2) = naturalGradientELBO_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invKmm,model.γ,stoch_coeff=model.StochCoeff,MBIndices=model.MBIndices,κ=model.κ)
    model.η_1 = grad_η_1; model.η_2 = grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = broadcast(x->-0.5*inv(x),model.η_2); model.μ = model.ζ.*model.η_1 #Back to the distribution parameters (needed for α updates)
end


function naturalGradientELBO_MultiClass(Y,θ_0,θ,invK,γ;stoch_coeff=1.0,MBIndices=0,κ=0)
    if κ == 0
        grad_1 = broadcast((y,gamma)->0.5*(y-gamma),Y,γ)
        grad_2 = broadcast((y,theta)->-0.5*(diagm(y.*θ_0+theta)+invK),Y,θ)
    else
        grad_1 = broadcast((y,kappa,gamma)->0.5*stoch_coeff*kappa'*(y[MBIndices]-gamma[MBIndices]),Y,κ,γ)
        grad_2 = broadcast((y,kappa,theta,invKmm)->-0.5*(stoch_coeff*kappa'*(diagm(y[MBIndices].*θ_0[MBIndices]+theta[MBIndices]))*kappa+invKmm),Y,κ,θ,invK)
    end
    return grad_1,grad_2
end
