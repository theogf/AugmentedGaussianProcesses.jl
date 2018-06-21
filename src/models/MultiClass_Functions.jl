function variablesUpdate_MultiClass!(model::MultiClass,iter)
    local_updates!(model)
    (model.η_1, model.η_2) = naturalGradientELBO_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invK,model.γ)
    global_updates!(model,nothing,nothing,iter)
end

function local_updates!(model::MultiClass)
    C = broadcast((var,m)->sqrt.(var.+m.^2),diag.(model.ζ),model.μ)
    model.θ[1] = [0.5./sqrt(C[model.y_class[i]][i])*tanh(0.5*C[model.y_class[i]][i]) for i in 1:model.nSamples ];
    # println(mean(model.α),broadcast(mean,model.γ))
    model.γ = broadcast((c,μ)->1.0./(2.0*model.β.*cosh.(0.5.*c)).*exp.(digamma.(model.α).-0.5.*μ),C,model.μ)
    model.α = [1+sum(broadcast(x->x[i],model.γ)) for i in 1:model.nSamples]
    model.θ[2:end] = broadcast((γ,c)->0.5.*γ./c.*tanh.(0.5.*c),model.γ,C)
end

function global_updates!(model::MultiClass,grad_1,grad_2)
    model.ζ = broadcast(x->-0.5*inv(x),model.η_2); model.μ = model.ζ.*model.η_1 #Back to the distribution parameters (needed for α updates)
end

function variablesUpdate_MultiClass!(model::SparseMultiClass,iter)
    local_updates!(model)
    (grad_η_1, grad_η_2) = naturalGradientELBO_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invKmm,model.γ,stoch_coeff=model.StochCoeff,MBIndices=model.MBIndices,κ=model.κ)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_updates!(model,grad_η_1,grad_η_2)
end

function local_updates!(model::SparseMultiClass)
    if model.KInducingPoints
        C = broadcast((m,var,kappa,ktilde)->sqrt.(ktilde+sum((kappa*var).*kappa,2)[:]+(kappa*m).^2),model.μ,model.ζ,model.κ,model.Ktilde)
        model.θ[1] = [0.5./C[model.y_class[i]][iter]*tanh(0.5*C[model.y_class[i]][iter]) for (iter,i) in enumerate(model.MBIndices) ];
        for l in 1:model.nInnerLoops
            model.γ = broadcast((c,kappa,μ)->0.5./(cosh.(0.5.*c).*model.β).*exp.(digamma.(model.α).-0.5.*kappa*μ),C,model.κ,model.μ)
            model.α = [1+sum(broadcast(x->x[i],model.γ)) for i in 1:model.nSamplesUsed]
        end
    else
        C = broadcast((m,var)->sqrt.(model.Ktilde[1]+sum((model.κ[1]*var).*model.κ[1],2)[:]+(model.κ[1]*m).^2),model.μ,model.ζ)
        model.θ[1] = [0.5./C[model.y_class[i]][iter]*tanh(0.5*C[model.y_class[i]][iter]) for (iter,i) in enumerate(model.MBIndices) ];
        for l in 1:model.nInnerLoops
            model.γ = broadcast((c,μ)->0.5./(cosh.(0.5.*c).*model.β).*exp.(digamma.(model.α).-0.5.*model.κ[1]*μ),C,model.μ)
            model.α = [1+sum(broadcast(x->x[i],model.γ)) for i in 1:model.nSamplesUsed]
        end
    end
    model.θ[2:end] = broadcast((γ,c)->0.5.*γ./c.*tanh.(0.5.*c),model.γ,C)
end

function global_updates!(model::SparseMultiClass,grad_1,grad_2)
    model.η_1 = (1.0-model.ρ_s).*model.η_1 + model.ρ_s.*grad_1; model.η_2 = (1.0-model.ρ_s).*model.η_2 + model.ρ_s.*grad_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = broadcast(x->-0.5*inv(x),model.η_2); model.μ = model.ζ.*model.η_1 #Back to the distribution parameters (needed for α updates)
end


function naturalGradientELBO_MultiClass(Y,θ_0,θ,invK,γ;stoch_coeff=1.0,MBIndices=0,κ=0)
    if κ == 0
        #No inducing points
        grad_1 = broadcast((y,gamma)->0.5*(y-gamma),Y,γ)
        grad_2 = broadcast((y,theta)->-0.5*(diagm(y.*θ_0+theta)+invK),Y,θ)
    elseif size(κ,1) == 1
        #Shared inducing points
        grad_1 = broadcast((y,gamma)->0.5*stoch_coeff*κ[1]'*(y[MBIndices]-gamma),Y,γ)
        grad_2 = broadcast((y,theta)->-0.5*(stoch_coeff*κ[1]'*(diagm(y[MBIndices].*θ_0+theta))*κ[1]+invK[1]),Y,θ)
    else
        #Varying inducing points
        grad_1 = broadcast((y,kappa,gamma)->0.5*stoch_coeff*kappa'*(y[MBIndices]-gamma),Y,κ,γ)
        grad_2 = broadcast((y,kappa,theta,invKmm)->-0.5*(stoch_coeff*kappa'*(diagm(y[MBIndices].*θ_0+theta))*kappa+invKmm),Y,κ,θ,invK)
    end
    return grad_1,grad_2
end


function ELBO(model::MultiClass)
    C = broadcast((var,m)->sqrt.(var.+m.^2),diag.(model.ζ),model.μ)
    ELBO_v = -model.nSamples*log(2.0)+0.5*model.K*size(model.X,2)+sum(model.α-log.(model.β)+log.(gamma.(model.α)))+dot(1-model.α,digamma.(model.α))
    ELBO_v += 0.5*sum(broadcast((y,gam,mu,theta,sigma)->logdet(model.invK)+logdet(sigma)+dot(y-gam,mu)-sum((Diagonal(y.*model.θ[1]+theta)+model.invK).*(sigma+mu*(mu'))),model.Y,model.γ,model.μ,model.θ[2:end],model.ζ))
    ELBO_v += sum(broadcast((gam,c,theta)->dot(gam,-log(2)-log.(gam)+1.0+digamma.(model.α)-log.(model.β))-sum(log.(cosh.(0.5*c)))+0.5*dot(c,c.*theta),model.γ,C,model.θ[2:end]))
    ELBO_v += sum([-log.(cosh.(0.5*C[model.y_class[i]][i]))+0.5*model.θ[1][i]*(C[model.y_class[i]][i]^2) for i in 1:model.nSamples])
    return -ELBO_v
end

function ELBO(model::SparseMultiClass)
    ELBO_v = -model.nSamples*log(2.0)+0.5*model.K*model.m+model.StochCoeff*(sum(model.α-log.(model.β)+log.(gamma.(model.α)))+dot(1-model.α,digamma.(model.α)))
    if model.KInducingPoints
        C = broadcast((var,m,kappa,ktilde)->sqrt.(ktilde+sum((kappa*var).*kappa,2)+(kappa*m).^2),model.ζ,model.μ,model.κ,model.Ktilde)
        ELBO_v += 0.5*sum(broadcast((y,gam,mu,theta,sigma,invK,kappa,ktilde)->model.StochCoeff*dot(y[model.MBIndices]-gam,kappa*mu)-
                      sum((model.StochCoeff*kappa'*Diagonal(y[model.MBIndices].*model.θ[1]+theta)*kappa+invK).*(sigma+mu*(mu')))-
                      model.StochCoeff*dot(y[model.MBIndices].*model.θ[1]+theta,ktilde),model.Y,model.γ,model.μ,model.θ[2:end],
                      model.ζ,model.invKmm,model.κ,model.Ktilde))
    else
        C = broadcast((var,m)->sqrt.(model.Ktilde[1]+sum((model.κ[1]*var).*model.κ[1],2)+(model.κ[1]*m).^2),model.ζ,model.μ)
        ELBO_v += 0.5*sum(broadcast((y,gam,mu,theta,sigma)->model.StochCoeff*dot(y[model.MBIndices]-gam,model.κ[1]*mu)-
                      sum((model.StochCoeff*model.κ[1]'*Diagonal(y[model.MBIndices].*model.θ[1]+theta)*model.κ[1]+model.invKmm[1]).*(sigma+mu*(mu')))-
                      model.StochCoeff*dot(y[model.MBIndices].*model.θ[1]+theta,model.Ktilde[1]),
                      model.Y,model.γ,model.μ,model.θ[2:end],model.ζ))
    end
    ELBO_v += 0.5*sum(logdet.(model.invKmm).+logdet.(model.ζ))
    ELBO_v += model.StochCoeff*sum(broadcast((gam,c,theta)->dot(gam,-log(2)-log.(gam)+1.0+digamma.(model.α)-log.(model.β))-sum(log.(cosh.(0.5*c)))+0.5*dot(c,c.*theta),model.γ,C,model.θ[2:end]))
    ELBO_v += model.StochCoeff*sum([-log.(cosh.(0.5*C[model.y_class[i]][iter]))+0.5*model.θ[1][iter]*(C[model.y_class[i]][iter]^2) for (iter,i) in enumerate(model.MBIndices)])
    return -ELBO_v
end

function hyperparameter_gradient_function(model::MultiClass)
    A = [model.invK*(model.ζ[i]+model.µ[i]*transpose(model.μ[i]))-eye(model.nSamples) for i in 1:model.K]
    return function(Js)
                V = model.invK*Js[1]
                return sum([0.5*sum(V.*transpose(A[i])) for i in 1:model.K])
            end
end

function hyperparameter_gradient_function(model::SparseMultiClass)
    #General values used for all gradients
    B = broadcast((mu,sigma)->mu*transpose(mu) + sigma,model.μ,model.ζ)
    Kmn = [kernelmatrix(model.inducingPoints[i],model.X[model.MBIndices,:],model.kernel) for i in 1:model.K]
    return function(Js)
        println(size(Js))
                Jmm = [x->Js[1][i] for i in 1:model.K]; Jnm = [x->Js[2][i] for i in 1:model.K]; Jnn = Js[3];
                println("Sizes")
                print(size(Jmm),size(Jnm),size(Jnn))
                ι = [(Jnm[i]-model.κ[i]*Jmm[i])*model.invKmm[i] for i in 1:model.K]
                Jtilde = [Jnn - sum(ι[i].*(Kmn[i].'),2) - sum(model.κ[i].*Jnm[i],2) for i in 1:model.K]
                V = model.invKmm.*Jmm
                return sum( broadcast((v,invK,iota,theta,kappa,b,jtilde,y,gam)->
                0.5*sum((v*invK-model.StochCoeff*(iota'*theta*kappa+kappa'*theta*iota)).*b')-trace(v)-model.StochCoeff*dot(model.θ[1].*y[model.MBIndices],jtilde)
                    + model.StochCoeff*(dot(y[model.MBIndices]-gam,iota*mu)),
                    V,model.invKmm,ι,model.θ[2:end],model.κ,B,Jtilde,model.y,model.γ)) #arugments
     end
end
