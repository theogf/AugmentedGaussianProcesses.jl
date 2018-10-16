#Set of functions for the multiclass model

"Update the local variational parameters of the full batch GP Multiclass"
function local_update!(model::MultiClass)
    model.f2 .= broadcast((Σ,μ)->sqrt.(Σ.+μ.^2),diag.(model.Σ[model.KIndices]),model.μ[model.KIndices])
    if model.KStochastic
        model.K_map = [findnext(x->x==model.y_class[i],model.KIndices,1) for i in 1:model.nSamples]
        model.θ[1] .= [model.K_map[i] != nothing ? 0.5./model.f2[model.K_map[i]][i]*tanh(0.5*model.f2[model.K_map[i]][i]) : 0 for i in 1:model.nSamples];
    else
        model.θ[1] .= [0.5./model.f2[model.y_class[i]][i]*tanh(0.5*model.f2[model.y_class[i]][i]) for i in 1:model.nSamples];
    end
    model.γ .= broadcast((f2,μ)->model.β./(2.0*gamma.(model.α).*cosh.(0.5.*f2)).*exp.(-model.α-(1.0.-model.α).*digamma.(model.α).-0.5.*μ),model.f2,model.μ[model.KIndices])
    model.α .= [1+model.KStochCoeff*sum([gam[i] for gam in model.γ]) for i in 1:model.nSamples]
    model.θ[2:end] = broadcast((γ,f2)->0.5.*γ./f2.*tanh.(0.5.*f2),model.γ,model.f2)
end

"Compute the variational updates for the full GP MultiClass"
function variational_updates!(model::MultiClass,iter::Integer)
    local_update!(model)
    natural_gradient_MultiClass!(model)
    global_update!(model)
end


"Update of the global variational parameter for full batch case"
function global_update!(model::MultiClass)
    model.Σ[model.KIndices] .= -inv.(model.η_2[model.KIndices]).*0.5;
    model.μ[model.KIndices] .= model.Σ[model.KIndices].*model.η_1[model.KIndices] #Back to the distribution parameters (needed for α updates)
end

"Return variational parameter gamma"
function local_gamma(f2::Vector{Float64},κ::Matrix{Float64},μ::Vector{Float64},β::Vector{Float64},α::Vector{Float64})
    return 0.5.*β./(cosh.(0.5.*f2).*gamma.(α)).*exp.(-α-(1.0.-α).*digamma.(α).-0.5.*κ*μ)
end

"Return the approximate variational parameter γ for large α"
function approx_local_gamma(f2::Vector{Float64},κ::Matrix{Float64},μ::Vector{Float64},β::Vector{Float64},α::Vector{Float64})
    return 0.5.*β./(cosh.(0.5.*f2)).*exp.(-(1.0.-α).*digamma.(α).-(α.-0.5).*log.(α).-0.5*log(2*pi).-0.5.*κ*μ)
end

"Compute the variational updates for the sparse GP XGPC"
function local_update!(model::SparseMultiClass)
    model.f2 = broadcast((μ::Vector{Float64},Σ::Symmetric{Float64,Matrix{Float64}},κ::Matrix{Float64},ktilde::Vector{Float64})->sqrt.(ktilde+sum((κ*Σ).*κ,dims=2)[:]+(κ*μ).^2),model.μ[model.KIndices],model.Σ[model.KIndices],model.κ,model.Ktilde)
    if model.KStochastic
        model.K_map = [findnext(x->x==model.y_class[i],model.KIndices,1) for i in model.MBIndices]
        model.θ[1] .= [model.K_map[i] != nothing ? 0.5./model.f2[model.K_map[i]][i]*tanh(0.5*model.f2[model.K_map[i]][i]) : 0 for i in 1:model.nSamplesUsed];
    else
        model.θ[1] .= [0.5./model.f2[model.y_class[i]][iter]*tanh(0.5*model.f2[model.y_class[i]][iter]) for (iter,i) in enumerate(model.MBIndices)];
    end
    for _ in 1:model.nInnerLoops
        if maximum(model.α) < 50

            model.γ .= broadcast(local_gamma,model.f2,model.κ,model.μ[model.KIndices],[model.β],[model.α[model.MBIndices]])
        else
            model.γ .= broadcast(approx_local_gamma,model.f2,model.κ,model.μ[model.KIndices],[model.β],[model.α[model.MBIndices]])
        end
        model.α[model.MBIndices] .= [1+model.KStochCoeff*sum([gam[i] for gam in model.γ]) for i in 1:model.nSamplesUsed]
    end
    model.θ[2:end] .= broadcast((γ::Vector{Float64},f2::Vector{Float64})->0.5.*γ./f2.*tanh.(0.5.*f2),model.γ,model.f2)
end

"Compute the variational updates for the sparse GP MultiClass"
function variational_updates!(model::SparseMultiClass,iter::Integer)
    local_update!(model)
    (grad_η_1, grad_η_2) = natural_gradient_MultiClass(model)
    # println("grad 1", [ mean(g) for g in grad_η_1])
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    # println("stochastic update", model.ρ_s)
    global_update!(model,grad_η_1,grad_η_2)
end


"""Update the global variational parameters for the sparse multiclass model"""
function global_update!(model::SparseMultiClass,grad_1::Vector{Vector{Float64}},grad_2::Vector{Matrix{Float64}})
    model.η_1[model.KIndices] .= (1.0.-model.ρ_s[model.KIndices]).*model.η_1[model.KIndices] + model.ρ_s[model.KIndices].*grad_1;
    model.η_2[model.KIndices] .= Symmetric.(model.η_2[model.KIndices].*(1.0.-model.ρ_s[model.KIndices]) + model.ρ_s[model.KIndices].*grad_2) #Update of the natural parameters with noisy/full natural gradient
    #TODO Temporary fix until LinearAlgebra has corrected it
    model.Σ[model.KIndices] .= -inv.(model.η_2[model.KIndices]).*0.5; model.μ[model.KIndices] .= model.Σ[model.KIndices].*model.η_1[model.KIndices] #Back to the distribution parameters (needed for α updates)
end

"""Compute the natural gradient of the ELBO given the natural parameters"""
function natural_gradient_MultiClass!(model::MultiClass)
        model.η_1[model.KIndices] .= broadcast((y,γ)->0.5*(y-γ),model.Y[model.KIndices],model.γ)
        model.η_2[model.KIndices] .= broadcast((y,θ,invK)->Symmetric(-0.5*(Diagonal(y.*model.θ[1]+θ)+invK)),model.Y[model.KIndices],model.θ[2:end],model.invK[model.KIndices])
end

"""Compute the natural gradient of the ELBO given the natural parameters"""
function natural_gradient_MultiClass(model::SparseMultiClass)
    grad_1 = broadcast((y::SparseVector{Int64,Int64},κ::Matrix{Float64},γ::Vector{Float64})->0.5*model.StochCoeff*transpose(κ)*Array(y[model.MBIndices]-γ),model.Y[model.KIndices],model.κ,model.γ)
    grad_2 = broadcast((y::SparseVector{Int64,Int64},κ::Matrix{Float64},θ::Vector{Float64},invKmm::Symmetric{Float64,Matrix{Float64}})->-0.5*(model.StochCoeff*κ'*Diagonal(y[model.MBIndices].*model.θ[1]+θ)*κ+invKmm),model.Y[model.KIndices],model.κ,model.θ[2:end],model.invKmm[model.IndependentGPs ? model.KIndices : :])
    return grad_1, grad_2
end


"""Return the negative ELBO for the MultiClass model"""
function ELBO(model::MultiClass)
    ELBO_v = model.nSamples*(0.5*model.K.-log(2))-sum(model.α./model.β)+sum(model.α-log.(model.β)+log.(gamma.(model.α))+(1.0.-model.α).*digamma.(model.α))
    if model.KStochastic
        ELBO_v += model.KStochCoeff*sum([model.K_map[i]!=nothing ? -log.(cosh.(0.5.*model.f2[model.K_map[i]][i]))+0.5*model.θ[1][i]*(model.f2[model.K_map[i]][i]^2) : 0 for i in 1:model.nSamples])
    else
        ELBO_v += sum([-log.(cosh.(0.5.*model.f2[model.y_class[i]][i]))+0.5*model.θ[1][i]*(model.f2[model.y_class[i]][i]^2) for i in 1:model.nSamples])
    end
    ELBO_v += 0.5*model.KStochCoeff*sum(broadcast((invK,y,gam,mu,theta,sigma)->logdet(invK)+logdet(sigma)+dot(y-gam,mu)-sum((Diagonal(y.*model.θ[1]+theta)+invK).*transpose(sigma+mu*(mu'))),model.invK[model.KIndices],model.Y[model.KIndices],model.γ,model.μ[model.KIndices],model.θ[2:end],model.Σ[model.KIndices]))
    ELBO_v += sum(broadcast((gam,f2,theta)->dot(gam,-(model.α.-log.(model.β).+log.(gamma.(model.α)).+(1.0.-model.α).*digamma.(model.α)).-log(2.0).-log.(gam).+1.0.-log.(cosh.(0.5.*f2))).+0.5*dot(f2,f2.*theta),model.γ,model.f2,model.θ[2:end]))
    return -ELBO_v
end

"""Return the negative ELBO for the sparse MultiClass model"""
function ELBO(model::SparseMultiClass)
    ELBO_v = -model.nSamples*log(2.0)+0.5*model.K*model.m+model.StochCoeff*(sum(model.α[model.MBIndices]-log.(model.β)+log.(gamma.(model.α[model.MBIndices])))+dot(1.0.-model.α[model.MBIndices],digamma.(model.α[model.MBIndices])))
    if model.IndependentGPs
        ELBO_v += 0.5*model.KStochCoeff*sum(broadcast((y,gam,mu,theta,sigma,invK,κ,ktilde)->model.StochCoeff*dot(y[model.MBIndices]-gam,κ*mu)-
                      sum((model.StochCoeff*κ'*Diagonal(y[model.MBIndices].*model.θ[1]+theta)*κ+invK).*(sigma+mu*(mu')))-
                      model.StochCoeff*dot(y[model.MBIndices].*model.θ[1]+theta,ktilde),model.Y[model.KIndices],model.γ,model.μ[model.KIndices],model.θ[2:end],
                      model.Σ[model.KIndices],model.invKmm[model.KIndices],model.κ,model.Ktilde))
        ELBO_v += 0.5*model.KStochCoeff*sum(logdet.(model.invKmm[model.KIndices]).+logdet.(model.Σ[model.KIndices]))
    else
        ELBO_v += 0.5*model.KStochCoeff*sum(broadcast((y,gam,mu,theta,sigma)->model.StochCoeff*dot(y[model.MBIndices]-gam,model.κ[1]*mu)-
                      sum((model.StochCoeff*model.κ[1]'*Diagonal(y[model.MBIndices].*model.θ[1]+theta)*model.κ[1]+model.invKmm[1]).*(sigma+mu*(mu')))-
                      model.StochCoeff*dot(y[model.MBIndices].*model.θ[1]+theta,model.Ktilde[1]),
                      model.Y[model.KIndices],model.γ,model.μ[model.KIndices],model.θ[2:end],model.Σ[model.KIndices]))
        ELBO_v += 0.5*(model.K*logdet(model.invKmm[1])+model.KStochCoeff*sum(logdet.(model.Σ[model.KIndices])))
    end
    ELBO_v += model.StochCoeff*model.KStochCoeff*sum(broadcast((gam,f2,theta)->dot(gam,-log(2).-log.(gam).+1.0.+digamma.(model.α[model.MBIndices])-log.(model.β))-sum(log.(cosh.(0.5*f2)))+0.5*dot(f2,f2.*theta),model.γ,model.f2,model.θ[2:end]))
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
        A = [model.invK[i]*(model.Σ[i]+model.µ[i]*model.μ[i]')-Diagonal{Float64}(I,model.nSamples) for i in model.KIndices]
        return (function(J,Kindex,index)
                    return 0.5*sum((model.invK[Kindex]*J).*transpose(A[index]))
                end,
                function(kernel,Kindex,index)
                    return 0.5/getvariance(kernel)*tr(A[index])
                end)
    else
        A = [model.invK[1]*(model.Σ[i]+model.µ[i]*model.μ[i]')-Diagonal{Float64}(I,model.nSamples) for i in model.KIndices]
        V = Matrix{Float64}(undef,model.nSamples,model.nSamples)
        return (function(J,Kindex,index)
            V = model.invK[1]*J #invK*J
            return 0.5*model.KStochCoeff*sum([sum(V.*transpose(A[i])) for i in 1:model.nClassesUsed])
                end,
                function(kernel)
                    return 0.5/getvariance(kernel)*sum(tr.(A))
                end)
    end
end

function add_transpose!(A::Matrix{T}) where {T}
    A .+= A'
end


"""Return the gradient of the ELBO given the kernel hyperparameters"""
function hyperparameter_gradient_function(model::SparseMultiClass)
    #General values used for all gradients
    F2 = broadcast((μ::Vector{Float64},Σ::Symmetric{Float64,Matrix{Float64}})->μ*transpose(μ) + Σ,model.μ[model.KIndices],model.Σ[model.KIndices])
    C = broadcast((y::SparseVector{Int64,Int64},θ::Vector{Float64})->Array(y[model.MBIndices].*model.θ[1]).+θ,model.Y[model.KIndices],model.θ[2:end])
    #preallocation
    ι = Matrix{Float64}(undef,model.nSamplesUsed,model.m)
    Jtilde = Vector{Float64}(undef,model.nSamplesUsed)
    V = Matrix{Float64}(undef,model.m,model.m)
    if model.IndependentGPs
        KC = transpose.(model.κ).*Diagonal.(C)
        A = Matrix{Float64}(undef,model.m,model.m)
        return (function(Jmm::LinearAlgebra.Symmetric{Float64,Matrix{Float64}},Jnm::Matrix{T},Jnn::Vector{T},Kindex::Int64,index::Int64) where {T}
                    # ι = (Jnm-model.κ[index]*Jmm)*model.invKmm[Kindex]
                    mul!(ι,(Jnm-model.κ[index]*Jmm),model.invKmm[Kindex])
                    # Jtilde = Jnn - sum(ι.*model.Knm[index],dims=2)[:] - sum(model.κ[index].*Jnm,dims=2)[:]
                    Jnn .+= - sum(ι.*model.Knm[index],dims=2)[:] - sum(model.κ[index].*Jnm,dims=2)[:]
                    # V = Matrix(model.invKmm[Kindex])*Matrix(Jmm)
                    mul!(V,model.invKmm[Kindex],Matrix(Jmm))
                    trV = tr(V)
                    V *= model.invKmm[Kindex]
                    A = add_transpose!(KC[index]*ι)
                    grad = sum(V.*F2[index])
                    grad += - model.StochCoeff*sum(A.*F2[index])
                    grad += - trV
                    grad += - model.StochCoeff*dot(C[index],Jnn)
                    grad += model.StochCoeff * dot(Array(model.Y[Kindex][model.MBIndices]) - model.γ[index], ι*model.μ[Kindex])
                    grad *= 0.5
                    return grad
                    # return 0.5*(sum((V*model.invKmm[Kindex]).*F2[index])-model.StochCoeff*sum(A.*F2[index])
                    #         - trV
                    #         - model.StochCoeff*dot(C[index],Jtilde)
                    #         + model.StochCoeff * dot(Array(model.Y[Kindex][model.MBIndices]) - model.γ[index], ι*model.μ[Kindex]))
         end, #end of function(Js)
                function(kernel::Kernel,Kindex::Int64,index::Int64)
                    return 0.5/(getvariance(kernel))*(sum(model.invKmm[Kindex].*F2[index])-model.StochCoeff * dot(C[index],model.Ktilde[index])-model.m)
                end)
    else
        KC = broadcast(c->transpose(model.κ[1])*Diagonal(c),C)
        A = [Matrix{Float64}(undef,model.m,model.m) for _ in 1:model.nClassesUsed]
        return (function(Jmm::LinearAlgebra.Symmetric{Float64,Matrix{Float64}},Jnm::Matrix{T},Jnn::Vector{T},Kindex::Int64,index::Int64) where {T}
                    mul!((Jnm-model.κ[1]*Jmm),model.invKmm[1])
                    Jnn .+= - sum(ι.*model.Knm[1],dims=2)[:] - sum(model.κ[1].*Jnm,dims=2)[:]
                    A .= broadcast(kc::Matrix{T}->add_transpose!(kc*ι),KC)
                    mul!(V,Matrix(model.invKmm[1]),Matrix(Jmm))
                    TraceV = -tr(V)
                    V*= model.invKmm[1]
                    return 0.5*(model.KStochCoeff*sum(broadcast((f2,a,c,y,γ,μ)->                (sum(V.*f2)-sum(a.*f2)*model.StochCoeff)
                        -model.StochCoeff*dot(c,Jtilde)
                        + model.StochCoeff*dot(Array(y[model.MBIndices])-γ,ι*μ),
                        F2,A,C,model.Y[model.KIndices],model.γ,model.μ[model.KIndices]))+model.K*TraceV)
        end,#end of function(Js)
        function(kernel::Kernel)
            # println(mean(F2[index]))
            return 0.5/(getvariance(kernel))*sum(broadcast((f2,c)->sum(model.invKmm[1].*f2)-model.StochCoeff * dot(c,model.Ktilde[1])-model.m,F2,C))
        end)
       end
end
