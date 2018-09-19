#Set of functions for the multiclass model

"Update the local variational parameters of the full batch GP Multiclass"
function local_update!(model::MultiClass)
    model.f2 .= broadcast((var,m)->sqrt.(var.+m.^2),diag.(model.Σ[model.KIndices]),model.μ[model.KIndices])
    model.θ[1] .= [K_map[iter] != nothing ? 0.5./model.f2[K_map[i]][i]*tanh(0.5*model.f2[K_map[i]][i]) : 0 for i in model.nSamples];
    if model.KStochastic
        model.K_map = [findnext(x->x==model.y_class[i],model.KIndices,1) for i in 1:model.nSamples]
        model.θ[1] .= [model.K_map[i] != nothing ? 0.5./model.f2[model.K_map[i]][i]*tanh(0.5*model.f2[model.K_map[i]][i]) : 0 for i in 1:model.nSamplesUsed];
    else
        model.θ[1] .= [0.5./model.f2[model.y_class[i]][i]*tanh(0.5*model.f2[model.y_class[i]][i]) for i in 1:model.nSamples];
    end
    model.γ .= broadcast((f2,μ)->model.β./(2.0*gamma.(model.α).*cosh.(0.5.*f2)).*exp.(-model.α-(1-model.α).*digamma.(model.α).-0.5.*μ),model.f2,model.μ[model.KIndices])
    model.α .= [1+model.KStochCoeff*sum([gam[i] for gam in model.γ]) for i in 1:model.nSamples]
    model.θ[2:end] = broadcast((γ,f2)->0.5.*γ./f2.*tanh.(0.5.*f2),model.γ,model.f2)
end

"Compute the variational updates for the full GP MultiClass"
function variational_updates!(model::MultiClass,iter::Integer)
    local_update!(model)
    (model.η_1[model.KIndices], model.η_2[model.KIndices]) = natural_gradient_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invK,model.γ,KKIndices=model.KIndices)
    global_update!(model)
end


"Update of the global variational parameter for full batch case"
function global_update!(model::MultiClass)
    model.Σ[model.KIndices] = broadcast(x->-0.5*inv(x),model.η_2[model.KIndices]);
    model.μ[model.KIndices] = model.Σ[model.KIndices].*model.η_1[model.KIndices] #Back to the distribution parameters (needed for α updates)
end


"Compute the variational updates for the sparse GP XGPC"
function local_update!(model::SparseMultiClass)
    if model.IndependentGPs
        model.f2 = broadcast((m,var,kappa,ktilde)->sqrt.(ktilde+sum((kappa*var).*kappa,dims=2)[:]+(kappa*m).^2),model.μ[model.KIndices],model.Σ[model.KIndices],model.κ[model.KIndices],model.Ktilde[model.KIndices])
        if model.KStochastic
            model.K_map = [findnext(x->x==model.y_class[i],model.KIndices,1) for i in model.MBIndices]
            model.θ[1] .= [model.K_map[i] != nothing ? 0.5./model.f2[model.K_map[i]][i]*tanh(0.5*model.f2[model.K_map[i]][i]) : 0 for i in 1:model.nSamplesUsed];
        else
            model.θ[1] .= [0.5./model.f2[model.y_class[i]][iter]*tanh(0.5*model.f2[model.y_class[i]][iter]) for (iter,i) in enumerate(model.MBIndices)];
        end
        for l in 1:model.nInnerLoops
            model.γ .= broadcast((f2,kappa,μ)->0.5.*model.β./(cosh.(0.5.*f2).*gamma.(model.α)).*exp.(-model.α-(1.0.-model.α).*digamma.(model.α).-0.5.*kappa*μ),model.f2,model.κ[model.KIndices],model.μ[model.KIndices])
            model.α .= [1+model.KStochCoeff*sum([gam[i] for gam in model.γ]) for i in 1:model.nSamplesUsed]
        end
    else
        model.f2 .= broadcast((m,var)->sqrt.(model.Ktilde[1]+sum((model.κ[1]*var).*model.κ[1],dims=2)[:]+(model.κ[1]*m).^2),model.μ[model.KIndices],model.Σ[model.KIndices])
        K_map = [findnextfindnext(x->x==model.y_class[i],model.KIndices,1) for i in model.MBIndices]
        model.θ[1] .= [K_map[i] != nothing ? 0.5./model.f2[K_map[i]][i]*tanh(0.5*model.f2[K_map[i]][i]) : 0 for i in model.MBIndices];
        for l in 1:model.nInnerLoops
            model.γ .= broadcast((f2,μ)->0.5.*model.β./(cosh.(0.5.*f2).*gamma.(model.α)).*exp.(-model.α-(1.0.-model.α).*digamma.(model.α).-0.5.*model.κ[1]*μ),model.f2,model.μ[model.KIndices])
            model.α .= [1+model.KStochCoeff*sum([gam[i] for gam in model.γ]) for i in 1:model.nSamplesUsed]
        end
    end
    model.θ[2:end] .= broadcast((γ,f2)->0.5.*γ./f2.*tanh.(0.5.*f2),model.γ,model.f2)
end

"Compute the variational updates for the sparse GP MultiClass"
function variational_updates!(model::SparseMultiClass,iter::Integer)
    local_update!(model)
    (grad_η_1, grad_η_2) = natural_gradient_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invKmm,model.γ,stoch_coeff=model.StochCoeff,MBIndices=model.MBIndices,κ=model.κ,KIndices=model.KIndices)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end


"""Update the global variational parameters for the sparse multiclass model"""
function global_update!(model::SparseMultiClass,grad_1::Array{Array{Float64,1},1},grad_2::Array{Array{Float64,2},1})
    model.η_1[model.KIndices] .= (1.0.-model.ρ_s[model.KIndices]).*model.η_1[model.KIndices] + model.ρ_s[model.KIndices].*grad_1; model.η_2[model.KIndices] .= (1.0.-model.ρ_s[model.KIndices]).*model.η_2[model.KIndices] + model.ρ_s[model.KIndices].*grad_2 #Update of the natural parameters with noisy/full natural gradient
    model.Σ[model.KIndices] .= -0.5.*inv.(model.η_2[model.KIndices]); model.μ[model.KIndices] .= model.Σ[model.KIndices].*model.η_1[model.KIndices] #Back to the distribution parameters (needed for α updates)
end

"""Compute the natural gradient of the ELBO given the natural parameters"""
function natural_gradient_MultiClass(Y::Vector{SparseVector{Int64}},θ_0::Vector{Float64},θ::Vector{Vector{Float64}},invK::Vector{Matrix{Float64}},γ::Vector{Vector{Float64}};stoch_coeff=1.0,MBIndices=0,κ=0,KIndices=0)
    if κ == 0
        #No shared inducing points
        grad_1 = broadcast((y,gamma)->0.5*(y-gamma),Y[KIndices],γ)
        grad_2 = broadcast((y,theta,invKnn)->-0.5*(diagm(y.*θ_0+theta)+invKnn),Y[KIndices],θ,invK[KIndices])
    elseif size(κ,1) == 1
        #Shared inducing points
        grad_1 = broadcast((y,gamma)->0.5*stoch_coeff*κ[1]'*(y[MBIndices]-gamma),Y[KIndices],γ)
        grad_2 = broadcast((y,theta)->-0.5*(stoch_coeff*κ[1]'*(Diagonal(y[MBIndices].*θ_0+theta))*κ[1]+invK[1]),Y[KIndices],θ)
    else
        #Varying inducing points
        grad_1 = broadcast((y,kappa,gamma)->0.5*stoch_coeff*kappa'*(y[MBIndices]-gamma),Y[KIndices],κ[KIndices],γ)
        grad_2 = broadcast((y,kappa,theta,invKmm)->-0.5*(stoch_coeff*kappa'*(Diagonal(y[MBIndices].*θ_0+theta))*kappa+invKmm),Y[KIndices],κ[KIndices],θ,invK[KIndices])
    end
    return grad_1,grad_2
end

"""Return the negative ELBO for the MultiClass model"""
function ELBO(model::MultiClass)
    ELBO_v = model.nSamples*(0.5*model.K-log(2))-sum(model.α./model.β)+sum(model.α-log.(model.β)+log.(gamma.(model.α))+(1-model.α).*digamma.(model.α))
    if model.KStochastic
        ELBO_v += model.KStochCoeff*sum([model.K_map[i]!=nothing ? -log.(cosh.(0.5*model.f2[model.K_map[i]][i]))+0.5*model.θ[1][i]*(model.f2[model.K_map[i]][i]^2) : 0 for i in 1:model.nSamples])
    else
        ELBO_v += sum([-log.(cosh.(0.5*model.f2[model.y_class[i]][i]))+0.5*model.θ[1][i]*(model.f2[model.y_class[i]][i]^2) for i in 1:model.nSamples])
    end
    ELBO_v += 0.5*model.KStochCoeff*sum(broadcast((invK,y,gam,mu,theta,sigma)->logdet(invK)+logdet(sigma)+dot(y-gam,mu)-sum((Diagonal(y.*model.θ[1]+theta)+invK).*transpose(sigma+mu*(mu'))),model.invK[model.KIndices],model.Y[model.KIndices],model.γ,model.μ[model.KIndices],model.θ[2:end],model.Σ[model.KIndices]))
    ELBO_v += sum(broadcast((gam,f2,theta)->dot(gam,-(model.α-log.(model.β)+log.(gamma.(model.α))+(1-model.α).*digamma.(model.α))-log(2.0)-log.(gam)+1.0-log.(cosh.(0.5*f2)))+0.5*dot(f2,f2.*theta),model.γ,model.f2,model.θ[2:end]))
    return -ELBO_v
end

"""Return the negative ELBO for the sparse MultiClass model"""
function ELBO(model::SparseMultiClass)
    ELBO_v = -model.nSamples*log(2.0)+0.5*model.K*model.m+model.StochCoeff*(sum(model.α-log.(model.β)+log.(gamma.(model.α)))+dot(1.0.-model.α,digamma.(model.α)))
    if model.IndependentGPs
        ELBO_v += 0.5*model.KStochCoeff*sum(broadcast((y,gam,mu,theta,sigma,invK,kappa,ktilde)->model.StochCoeff*dot(y[model.MBIndices]-gam,kappa*mu)-
                      sum((model.StochCoeff*kappa'*Diagonal(y[model.MBIndices].*model.θ[1]+theta)*kappa+invK).*(sigma+mu*(mu')))-
                      model.StochCoeff*dot(y[model.MBIndices].*model.θ[1]+theta,ktilde),model.Y[model.KIndices],model.γ,model.μ[model.KIndices],model.θ[2:end],
                      model.Σ[model.KIndices],model.invKmm[model.KIndices],model.κ[model.KIndices],model.Ktilde[model.KIndices]))
    else
        ELBO_v += 0.5*model.KStochCoeff*sum(broadcast((y,gam,mu,theta,sigma)->model.StochCoeff*dot(y[model.MBIndices]-gam,model.κ[1]*mu)-
                      sum((model.StochCoeff*model.κ[1]'*Diagonal(y[model.MBIndices].*model.θ[1]+theta)*model.κ[1]+model.invKmm[1]).*(sigma+mu*(mu')))-
                      model.StochCoeff*dot(y[model.MBIndices].*model.θ[1]+theta,model.Ktilde[1]),
                      model.Y[model.KIndices],model.γ,model.μ[model.KIndices],model.θ[2:end],model.Σ[model.KIndices]))
    end
    ELBO_v += 0.5*model.KStochCoeff*sum(logdet.(model.invKmm[model.KIndices]).+logdet.(model.Σ[model.KIndices]))
    ELBO_v += model.StochCoeff*model.KStochCoeff*sum(broadcast((gam,f2,theta)->dot(gam,-log(2).-log.(gam).+1.0.+digamma.(model.α)-log.(model.β))-sum(log.(cosh.(0.5*f2)))+0.5*dot(f2,f2.*theta),model.γ,model.f2,model.θ[2:end]))
    if model.KStochastic
        ELBO_v += model.StochCoeff*model.KStochCoeff*sum([model.K_map[i] != nothing ? -log.(cosh.(0.5*model.f2[model.K_map[i]][i]))-0.5*model.θ[1][i]*(model.f2[model.K_map[i]][i]^2) : 0 for i in 1:model.nSamplesUsed])
    else
        ELBO_v += model.StochCoeff*sum([-log.(cosh.(0.5*model.f2[model.y_class[i]][iter]))-0.5*model.θ[1][iter]*(model.f2[model.y_class[i]][iter]^2) for (iter,i) in enumerate(model.MBIndices)])
    end
    return -ELBO_v
end

"""Return the gradient of the ELBO given the kernel hyperparameters"""
function hyperparameter_gradient_function(model::MultiClass)
    if model.IndependentGPs
        A = [model.invK[i]*(model.Σ[i]+model.µ[i]*model.μ[i]')-Diagonal{Float64}(I,model.nSamples) for i in 1:model.K]
        return function(Js,index)
                    V = model.invK[index].*Js[1] #invK*Js
                    return 0.5*sum(V.*transpose(A[index]))
                end
    else
        A = [model.invK[1]*(model.Σ[i]+model.µ[i]*model.μ[i]')-Diagonal{Float64}(I,model.nSamples) for i in 1:model.K]
        return function(Js,index)
            V = matrices[1].*Js[1] #invK*Js
            return 0.5*sum([sum(V.*transpose(A[i])) for i in 1:model.K])
        end
    end
end

"""Return the gradient of the ELBO given the kernel hyperparameters"""
function hyperparameter_gradient_function(model::SparseMultiClass)
    #General values used for all gradients
    B = broadcast((mu,sigma)->mu*transpose(mu) + sigma,model.μ,model.Σ)
    if model.IndependentGPs
        Kmn = [kernelmatrix(model.inducingPoints[i],model.X[model.MBIndices,:],model.kernel[i]) for i in 1:model.K]
        return function(Js,index)
                    Jmm = Js[1]; Jnm =Js[2]; Jnn = Js[3];
                    ι = (Jnm-model.κ[index]*Jmm)*model.invKmm[index]
                    Jtilde = Jnn - sum(ι.*transpose(Kmn[index]),dims=2) - sum(model.κ[index].*Jnm,dims=2)
                    V = model.invKmm[index]*Jmm
                    # println("$index, $(mean(diag(V*model.invKmm[index])))")
                    # println("$index, $(sum((V*model.invKmm[index]).*B[index]))")
                    return 0.5*sum( (V*model.invKmm[index]
                            -model.StochCoeff*(ι'*Diagonal(model.θ[index+1])*model.κ[index]
                                                - model.κ[index]'*Diagonal(model.θ[index+1])*ι)).*B[index]')
                            -tr(V)-model.StochCoeff*dot(model.θ[1].*model.Y[index][model.MBIndices],Jtilde)
                            + model.StochCoeff*dot(model.Y[index][model.MBIndices]-model.γ[index],ι*model.μ[index])
         end #end of function(Js)
    else
        Kmn = kernelmatrix(model.inducingPoints[1],model.X[model.MBIndices,:],model.kernel[1])
        return function(Js,index)
            #matrices L: [1]Kmm, [2]invKmm, [3]κ
                    Jmm = Js[1]; Jnm = Js[2]; Jnn = Js[3];
                    ι = (Jnm-model.κ[1]*Jmm)/model.Kmm[1]
                    Jtilde = Jnn - sum(ι.*transpose(Kmn),dims=2) - sum(model.κ[1].*Jnm,dims=2)
                    V = model.Kmm[1]\Jmm
                    return sum(broadcast((theta,b,y,gam,mu)->
                        0.5*(sum((V/model.Kmm[1]-model.StochCoeff*(ι'*theta*model.κ[1]+model.κ[1]'*theta*ι)).*b')
                        -tr(V)-model.StochCoeff*dot(model.θ[1].*y[model.MBIndices],Jtilde)
                        + model.StochCoeff*dot(y[model.MBIndices]-gam,ι*mu)),
                        Diagonal.(model.θ[2:end]),B,model.Y,model.γ,model.μ)) #arguments
        end#end of function(Js)
    end
end
