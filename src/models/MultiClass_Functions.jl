function variablesUpdate_MultiClass!(model::MultiClass,iter)
    expec_f2 = broadcast((var,m)->sqrt.(var.+m.^2),diag.(model.ζ),model.μ)
    model.θ[1] = [0.5./sqrt(expec_f2[model.y_class[i]][i])*tanh(0.5*expec_f2[model.y_class[i]][i]) for i in 1:model.nSamples ];
    # println(mean(model.α),broadcast(mean,model.γ))
    model.γ = broadcast((f,μ)->1.0./(2.0*model.β.*cosh.(0.5.*f)).*exp.(digamma.(model.α).-0.5.*μ),expec_f2,model.μ)
    model.α = [1+sum(broadcast(x->x[i],model.γ)) for i in 1:model.nSamples]
    model.θ[2:end] = broadcast((γ,f)->0.5.*γ./f.*tanh.(0.5.*f),model.γ,expec_f2)
    (model.η_1, model.η_2) = naturalGradientELBO_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invK,model.γ)
    model.ζ = broadcast(x->-0.5*inv(x),model.η_2); model.μ = model.ζ.*model.η_1 #Back to the distribution parameters (needed for α updates)
end

function variablesUpdate_MultiClass!(model::SparseMultiClass,iter)
    expec_f2 = broadcast((m,var,kappa,ktilde)->sqrt.(ktilde+sum((kappa*var).*kappa,2)+(kappa*m).^2),model.μ,model.ζ,model.κ,model.Ktilde)
    model.θ[1][model.MBIndices] = [0.5./expec_f2[model.y_class[i]][i]*tanh(0.5*expec_f2[model.y_class[i]][i]) for i in model.MBIndices ];
    # println(mean(model.α),broadcast(mean,model.γ))
    broadcast((gamma,f,kappa,μ)->gamma[model.MBIndices]=0.5./(cosh.(0.5.*f).*model.β[model.MBIndices]).*exp.(digamma.(model.α[model.MBIndices]).-0.5.*kappa*μ),model.γ,expec_f2,model.κ,model.μ)
    model.α[model.MBIndices] = [1+sum(broadcast(x->x[i],model.γ)) for i in model.MBIndices]
    broadcast((theta,γ,f)->theta[model.MBIndices]=0.5.*γ./f.*tanh.(0.5.*f),model.θ[2:end],model.γ,expec_f2)
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


function ELBO(model::MultiClass)
    expec_f2 = broadcast((var,m)->sqrt.(var.+m.^2),diag.(model.ζ),model.μ)
    ELBO_v = -model.nSamples*log(2.0)+0.5*model.K*size(model.X,2)+sum(model.α-log.(model.β)+log.(gamma.(model.α)))+dot(1-model.α,digamma.(model.α))
    ELBO_v += 0.5*sum(broadcast((y,gam,mu,theta,sigma)->logdet(model.invK)+logdet(sigma)+dot(y-gam,mu)-sum((Diagonal(y.*model.θ[1]+theta)+model.invK).*(sigma+mu*(mu'))),broadcast(x->x[model.MBIndices],model.Y),broadcast(x->x[model.MBIndices],model.γ),model.μ,broadcast(x->x[model.MBIndices],model.θ[2:end]),model.ζ,model.κ))
    ELBO_v += sum(broadcast((gam,f2,theta)->dot(gam,-log(2)-log.(gam)+1+digamma.(model.α)-log.(model.β))-sum(log.(cosh.(0.5*f2)))+0.5*dot(f2,f2.*theta),model.γ,expec_f2,model.θ[2:end]))
    ELBO_v += sum([-log.(cosh.(0.5*expec_f2[model.y_class[i]][i]))+0.5*model.θ[1][i]*(expec_f2[model.y_class[i]][i]^2) for i in 1:model.nSamples])
    return -ELBO_v
end

function ELBO(model::SparseMultiClass)
    expec_f2 = broadcast((var,m)->sqrt.(var.+m.^2),diag.(model.ζ),model.μ)
    ELBO_v = -model.nSamples*log(2.0)+0.5*model.K*model.m+sum(model.α-log.(model.β)+log.(gamma.(model.α)))+dot(1-model.α,digamma.(model.α))
    ELBO_v += 0.5*sum(broadcast((y,gam,mu,theta,sigma,kappa)->logdet(model.invK)+logdet(sigma)+dot(y-gam,kappa*mu)-sum((kappa'*Diagonal(y.*model.θ[1]+theta)*kappa+invK).*(sigma+mu*(mu'))),model.Y,model.γ,model.μ,model.θ[2:end],model.ζ))
    ELBO_v += sum(broadcast((gam,f2,theta)->dot(gam,-log(2)-log.(gam)+1+digamma.(model.α)-log.(model.β))-sum(log.(cosh.(0.5*f2)))+0.5*dot(f2,f2.*theta),model.γ,expec_f2,model.θ[2:end]))
    model.θ[1] = [0.5./sqrt(expec_f2[model.y_class[i]][i])*tanh(0.5*expec_f2[model.y_class[i]][i]) for i in 1:model.nSamples ];
    ELBO_v += sum([-log.(cosh.(0.5*expec_f2[model.y_class[i]][i]))+0.5*model.θ[1][i]*(expec_f2[model.y_class[i]][i]^2) for i in 1:model.nSamples])
    return -ELBO_v
end
