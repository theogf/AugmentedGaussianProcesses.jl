#Set of functions for the multiclass model

"Update the local variational parameters of the full batch GP Multiclass"
function local_update!(model::MultiClass)
    C = broadcast((var,m)->sqrt.(var.+m.^2),diag.(model.Σ),model.μ)
    model.θ[1] = [0.5./sqrt(C[model.y_class[i]][i])*tanh(0.5*C[model.y_class[i]][i]) for i in 1:model.nSamples ];
    for i in 1:model.nInnerLoops
        model.γ = broadcast((c,μ)->model.β./(2.0*gamma.(model.α).*cosh.(0.5.*c)).*exp.(-model.α-(1-model.α).*digamma.(model.α).-0.5.*μ),C,model.μ)
        model.α = [1+sum(broadcast(x->x[i],model.γ)) for i in 1:model.nSamples]
    end
    model.θ[2:end] = broadcast((γ,c)->0.5.*γ./c.*tanh.(0.5.*c),model.γ,C)
end

"Compute the variational updates for the full GP MultiClass"
function variational_updates!(model::MultiClass,iter::Integer)
    local_update!(model)
    (model.η_1, model.η_2) = natural_gradient_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invK,model.γ)
    global_update!(model)
end


"Update of the global variational parameter for full batch case"
function global_update!(model::MultiClass)
    model.Σ = broadcast(x->-0.5*inv(x),model.η_2);
    model.μ = model.Σ.*model.η_1 #Back to the distribution parameters (needed for α updates)
end

"Compute the variational updates for the sparse GP XGPC"
function local_update!(model::SparseMultiClass)
    if model.IndependentGPs
        C = broadcast((m,var,kappa,ktilde)->sqrt.(ktilde+sum((kappa*var).*kappa,dims=2)[:]+(kappa*m).^2),model.μ,model.Σ,model.κ,model.Ktilde)
        model.θ[1] = [0.5./C[model.y_class[i]][iter]*tanh(0.5*C[model.y_class[i]][iter]) for (iter,i) in enumerate(model.MBIndices) ];
        for l in 1:model.nInnerLoops
            model.γ = broadcast((c,kappa,μ)->0.5.*model.β./(cosh.(0.5.*c).*gamma.(model.α)).*exp.(-model.α-(1.0.-model.α).*digamma.(model.α).-0.5.*kappa*μ),C,model.κ,model.μ)
            model.α = [1+sum(broadcast(x->x[i],model.γ)) for i in 1:model.nSamplesUsed]
        end
    else
        C = broadcast((m,var)->sqrt.(model.Ktilde[1]+sum((model.κ[1]*var).*model.κ[1],dims=2)[:]+(model.κ[1]*m).^2),model.μ,model.Σ)
        model.θ[1] = [0.5./C[model.y_class[i]][iter]*tanh(0.5*C[model.y_class[i]][iter]) for (iter,i) in enumerate(model.MBIndices) ];
        for l in 1:model.nInnerLoops
            model.γ = broadcast((c,μ)->0.5.*model.β./(cosh.(0.5.*c).*gamma.(model.α)).*exp.(-model.α-(1-model.α).*digamma.(model.α).-0.5.*model.κ[1]*μ),C,model.μ)
            model.α = [1+sum(broadcast(x->x[i],model.γ)) for i in 1:model.nSamplesUsed]
        end
    end
    model.θ[2:end] = broadcast((γ,c)->0.5.*γ./c.*tanh.(0.5.*c),model.γ,C)
end

"Compute the variational updates for the sparse GP MultiClass"
function variablesUpdate_MultiClass!(model::SparseMultiClass,iter::Integer)
    local_update!(model)
    (grad_η_1, grad_η_2) = natural_gradient_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invKmm,model.γ,stoch_coeff=model.StochCoeff,MBIndices=model.MBIndices,κ=model.κ)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end


"""Update the global variational parameters for the sparse multiclass model""" global_update!(model::SparseMultiClass,grad_1::Array{Array{Float64,1},1},grad_2::Array{Array{Float64,2},1})
function
    model.η_1 = (1.0.-model.ρ_s).*model.η_1 + model.ρ_s.*grad_1; model.η_2 = (1.0.-model.ρ_s).*model.η_2 + model.ρ_s.*grad_2 #Update of the natural parameters with noisy/full natural gradient
    model.Σ = broadcast(x->-0.5*inv(x),model.η_2); model.μ = model.Σ.*model.η_1 #Back to the distribution parameters (needed for α updates)
end

"""Compute the natural gradient of the ELBO given the natural parameters"""
function natural_gradient_MultiClass(Y::Array{Array{Float64,1},1},θ_0::Vector{Float64},θ::Vector{Vector},invK::Vector{Matrix},γ::Vector{Vector};stoch_coeff=1.0,MBIndices=0,κ=0)
    if κ == 0
        #No shared inducing points
        grad_1 = broadcast((y,gamma)->0.5*(y-gamma),Y,γ)
        grad_2 = broadcast((y,theta,invKnn)->-0.5*(diagm(y.*θ_0+theta)+invKnn),Y,θ,invK)
    elseif size(κ,1) == 1
        #Shared inducing points
        grad_1 = broadcast((y,gamma)->0.5*stoch_coeff*κ[1]'*(y[MBIndices]-gamma),Y,γ)
        grad_2 = broadcast((y,theta)->-0.5*(stoch_coeff*κ[1]'*(diagm(y[MBIndices].*θ_0+theta))*κ[1]+invK[1]),Y,θ)
    else
        #Varying inducing points
        grad_1 = broadcast((y,kappa,gamma)->0.5*stoch_coeff*kappa'*(y[MBIndices]-gamma),Y,κ,γ)
        grad_2 = broadcast((y,kappa,theta,invKmm)->-0.5*(stoch_coeff*kappa'*(Diagonal(y[MBIndices].*θ_0+theta))*kappa+invKmm),Y,κ,θ,invK)
    end
    return grad_1,grad_2
end

"""Return the negative ELBO for the MultiClass model"""
function ELBO(model::MultiClass)
    C = broadcast((var,m)->sqrt.(var.+m.^2),diag.(model.Σ),model.μ)
    ELBO_v = model.nSamples*(0.5*model.K-log(2))-sum(model.α./model.β)+sum(model.α-log.(model.β)+log.(gamma.(model.α))+(1-model.α).*digamma.(model.α))
    ELBO_v += sum([-log.(cosh.(0.5*C[model.y_class[i]][i]))+0.5*model.θ[1][i]*(C[model.y_class[i]][i]^2) for i in 1:model.nSamples])
    ELBO_v += 0.5*sum(broadcast((invK,y,gam,mu,theta,sigma)->logdet(invK)+logdet(sigma)+dot(y-gam,mu)-sum((Diagonal(y.*model.θ[1]+theta)+invK).*transpose(sigma+mu*(mu'))),model.invK,model.Y,model.γ,model.μ,model.θ[2:end],model.Σ))
    ELBO_v += sum(broadcast((gam,c,theta)->dot(gam,-(model.α-log.(model.β)+log.(gamma.(model.α))+(1-model.α).*digamma.(model.α))-log(2.0)-log.(gam)+1.0-log.(cosh.(0.5*c)))+0.5*dot(c,c.*theta),model.γ,C,model.θ[2:end]))
    return -ELBO_v
end

"""Return the negative ELBO for the sparse MultiClass model"""
function ELBO(model::SparseMultiClass)
    ELBO_v = -model.nSamples*log(2.0)+0.5*model.K*model.m+model.StochCoeff*(sum(model.α-log.(model.β)+log.(gamma.(model.α)))+dot(1.0.-model.α,digamma.(model.α)))
    if model.IndependentGPs
        C = broadcast((var,m,kappa,ktilde)->sqrt.(ktilde+sum((kappa*var).*kappa,dims=2)+(kappa*m).^2),model.Σ,model.μ,model.κ,model.Ktilde)
        ELBO_v += 0.5*sum(broadcast((y,gam,mu,theta,sigma,invK,kappa,ktilde)->model.StochCoeff*dot(y[model.MBIndices]-gam,kappa*mu)-
                      sum((model.StochCoeff*kappa'*Diagonal(y[model.MBIndices].*model.θ[1]+theta)*kappa+invK).*(sigma+mu*(mu')))-
                      model.StochCoeff*dot(y[model.MBIndices].*model.θ[1]+theta,ktilde),model.Y,model.γ,model.μ,model.θ[2:end],
                      model.Σ,model.invKmm,model.κ,model.Ktilde))
    else
        C = broadcast((var,m)->sqrt.(model.Ktilde[1]+sum((model.κ[1]*var).*model.κ[1],dims=2)+(model.κ[1]*m).^2),model.Σ,model.μ)
        ELBO_v += 0.5*sum(broadcast((y,gam,mu,theta,sigma)->model.StochCoeff*dot(y[model.MBIndices]-gam,model.κ[1]*mu)-
                      sum((model.StochCoeff*model.κ[1]'*Diagonal(y[model.MBIndices].*model.θ[1]+theta)*model.κ[1]+model.invKmm[1]).*(sigma+mu*(mu')))-
                      model.StochCoeff*dot(y[model.MBIndices].*model.θ[1]+theta,model.Ktilde[1]),
                      model.Y,model.γ,model.μ,model.θ[2:end],model.Σ))
    end
    ELBO_v += 0.5*sum(logdet.(model.invKmm).+logdet.(model.Σ))
    ELBO_v += model.StochCoeff*sum(broadcast((gam,c,theta)->dot(gam,-log(2).-log.(gam).+1.0.+digamma.(model.α)-log.(model.β))-sum(log.(cosh.(0.5*c)))+0.5*dot(c,c.*theta),model.γ,C,model.θ[2:end]))
    ELBO_v += model.StochCoeff*sum([-log.(cosh.(0.5*C[model.y_class[i]][iter]))-0.5*model.θ[1][iter]*(C[model.y_class[i]][iter]^2) for (iter,i) in enumerate(model.MBIndices)])
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
