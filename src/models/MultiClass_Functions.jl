#Set of functions for the multiclass model

"Update the local variational parameters of the full batch GP Multiclass"
function local_update!(model::MultiClass)
    model.c .= broadcast((Σ,μ)->sqrt.(Σ.+μ.^2),diag.(model.Σ[model.KIndices]),model.μ[model.KIndices])
    for _ in 1:2
        model.γ .= broadcast((c,μ)->0.5./(model.β.*cosh.(0.5.*c)).*exp.(digamma.(model.α).-0.5.*μ),model.c,model.μ[model.KIndices])
        model.α .= [1.0+model.KStochCoeff*sum([γ[i] for γ in model.γ]) for i in 1:model.nSamples]
    end
    model.θ .= broadcast((y,γ,c)->0.5.*Array(y+γ)./c.*tanh.(0.5.*c),model.Y[model.KIndices],model.γ,model.c)
end

"Compute the variational updates for the full GP MultiClass"
function variational_updates!(model::MultiClass,iter::Integer)
    local_update!(model)
    natural_gradient!(model)
    global_update!(model)
end


"Update of the global variational parameter for full batch case"
function global_update!(model::MultiClass)
    model.Σ[model.KIndices] .= -inv.(model.η_2[model.KIndices]).*0.5;
    model.μ[model.KIndices] .= model.Σ[model.KIndices].*model.η_1[model.KIndices] #Back to the distribution parameters (needed for α updates)
end

"Compute the variational updates for the sparse GP XGPC"
function local_update!(model::SparseMultiClass{T}) where T
    model.c = broadcast((μ::Vector{T},Σ::Symmetric{T,Matrix{T}},κ::Matrix{T},ktilde::Vector{T})->sqrt.(ktilde+vec(sum((κ*Σ).*κ,dims=2))+(κ*μ).^2),model.μ[model.KIndices],model.Σ[model.KIndices],model.κ,model.Ktilde)
    for _ in 1:model.nInnerLoops
        model.γ .= broadcast((c,κ,μ)->0.5./(model.β.*cosh.(0.5.*c)).*exp.(digamma.(model.α[model.MBIndices]).-0.5.*κ*μ),model.c,model.κ,model.μ[model.KIndices])
        model.α[model.MBIndices] .= [1.0+model.KStochCoeff*sum(γ[i] for γ in model.γ) for i in 1:model.nSamplesUsed]
    end
    model.θ .= broadcast((y,γ::Vector{T},c::Vector{T})->0.5.*Array(y[model.MBIndices]+γ)./c.*tanh.(0.5.*c),model.Y[model.KIndices],model.γ,model.c)
end

"Compute the variational updates for the sparse GP MultiClass"
function variational_updates!(model::SparseMultiClass,iter::Integer)
    local_update!(model)
    (grad_η_1, grad_η_2) = natural_gradient(model)
    # println("grad 1", [ mean(g) for g in grad_η_1])
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    # println("stochastic update", model.ρ_s)
    global_update!(model,grad_η_1,grad_η_2)
end


"""Update the global variational parameters for the sparse multiclass model"""
function global_update!(model::SparseMultiClass,grad_1::Vector{Vector{T}},grad_2::Vector{Matrix{T}}) where T
    model.η_1[model.KIndices] .= (1.0.-model.ρ_s[model.KIndices]).*model.η_1[model.KIndices] + model.ρ_s[model.KIndices].*grad_1;
    model.η_2[model.KIndices] .= Symmetric.(model.η_2[model.KIndices].*(1.0.-model.ρ_s[model.KIndices]) + model.ρ_s[model.KIndices].*grad_2) #Update of the natural parameters with noisy/full natural gradient
    #TODO Temporary fix until LinearAlgebra has corrected it
    model.Σ[model.KIndices] .= -inv.(model.η_2[model.KIndices]).*0.5; model.μ[model.KIndices] .= model.Σ[model.KIndices].*model.η_1[model.KIndices] #Back to the distribution parameters (needed for α updates)
end

"""Compute the natural gradient of the ELBO given the natural parameters"""
function natural_gradient!(model::MultiClass{T}) where T
        model.η_1[model.KIndices] .= broadcast((y,γ)->0.5*Array(y-γ),model.Y[model.KIndices],model.γ)
        if model.IndependentGPs
            model.η_2[model.KIndices] .= broadcast((y,θ,invK)->Symmetric(-0.5*(Diagonal(θ)+invK)),model.Y[model.KIndices],model.θ,model.invK[model.KIndices])
        else
            model.η_2[model.KIndices] .= broadcast((y,θ)->Symmetric(-0.5*(Diagonal(θ)+model.invK[1])),model.Y[model.KIndices],model.θ)
        end
end

"""Compute the natural gradient of the ELBO given the natural parameters"""
function natural_gradient(model::SparseMultiClass{T}) where T
    grad_1 = broadcast((y::SparseVector{Int64,Int64},κ::Matrix{T},γ::Vector{T})->0.5*model.StochCoeff*κ'*Array(y[model.MBIndices]-γ),model.Y[model.KIndices],model.κ,model.γ)
    grad_2 = broadcast((y::SparseVector{Int64,Int64},κ::Matrix{T},θ::Vector{T},invKmm::Symmetric{T,Matrix{T}})->-0.5*(model.StochCoeff*κ'*Diagonal(θ)*κ+invKmm),model.Y[model.KIndices],model.κ,model.θ,model.invKmm[model.IndependentGPs ? model.KIndices : :])
    return grad_1, grad_2
end


"""Return the negative ELBO for the MultiClass model"""
function ELBO(model::MultiClass)
    model.c .= broadcast((Σ,μ)->sqrt.(Σ.+μ.^2),diag.(model.Σ[model.KIndices]),model.μ[model.KIndices])
    ELBO_v = 0.0
    ELBO_v += model.KStochCoeff*ExpecLogLikelihood(model)
    ELBO_v += -model.KStochCoeff*GaussianKL(model)
    ELBO_v += -GammaImproperKL(model)
    ELBO_v += -model.KStochCoeff*PoissonKL(model)
    ELBO_v += -model.KStochCoeff*PolyaGammaKL(model)
    return -ELBO_v
end

"""Return the negative ELBO for the Sparse MultiClass model"""
function ELBO(model::SparseMultiClass)
    model.c = broadcast((μ::AbstractVector,Σ::AbstractMatrix,κ::AbstractMatrix,ktilde::AbstractVector)->sqrt.(ktilde+vec(sum((κ*Σ).*κ,dims=2))+(κ*μ).^2),model.μ[model.KIndices],model.Σ[model.KIndices],model.κ,model.Ktilde)
    ELBO_v = 0.0
    ELBO_v +=  model.KStochCoeff*model.StochCoeff*ExpecLogLikelihood(model)
    ELBO_v += -model.KStochCoeff*GaussianKL(model)
    ELBO_v += -GammaImproperKL(model)
    ELBO_v += -model.KStochCoeff*model.StochCoeff*PoissonKL(model)
    ELBO_v += -model.KStochCoeff*model.StochCoeff*PolyaGammaKL(model)
    return -ELBO_v
end

"""Return the negative ELBO for the Sparse MultiClass model"""
function ELBO(model::Union{SoftMaxMultiClass,LogisticSoftMaxMultiClass,SparseSoftMaxMultiClass,SparseLogisticSoftMaxMultiClass})
    ELBO_v = 0.0
    ELBO_v += ExpecLogLikelihood(model)
    ELBO_v -= GaussianKL(model)
    return -ELBO_v
end

function ExpecLogLikelihood(model::MultiClass)
    tot = 0.0
    tot += -model.nSamples*log(2.0)
    tot += -sum(sum.(model.γ[model.KIndices]))*log(2.0)
    tot += 0.5*sum(broadcast((y,μ,γ,θ,c)->sum(μ.*Array(y-γ)-θ.*(c.^2)),model.Y[model.KIndices],model.μ[model.KIndices],model.γ,model.θ,model.c))
end

function ExpecLogLikelihood(model::SparseMultiClass)
    tot = 0.0
    tot += -model.nSamplesUsed*log(2.0)
    tot += -sum(sum.(model.γ))*log(2.0)
    tot += 0.5*sum(broadcast((y,κ,μ,γ,θ,c)->sum((κ*μ).*Array(y[model.MBIndices]-γ)-θ.*(c.^2)),model.Y[model.KIndices],model.κ,model.μ[model.KIndices],model.γ,model.θ,model.c))
end

function ExpecLogLikelihood(model::SoftMaxMultiClass)
    tot = 0.0
    nSamples = 200
    for i in 1:model.nSamples
        p = MvNormal([model.μ[k][i] for k in 1:model.K],[sqrt(model.Σ[k][i,i]) for k in 1:model.K])
        class = model.ind_mapping[model.y[i]]
        for _ in 1:nSamples
            tot += log(softmax(rand(p),class))/nSamples
        end
    end
    return tot
end

function ExpecLogLikelihood(model::LogisticSoftMaxMultiClass)
    tot = 0.0
    nSamples = 200
    for i in 1:model.nSamples
        p = MvNormal([model.μ[k][i] for k in 1:model.K],[sqrt(model.Σ[k][i,i]) for k in 1:model.K])
        class = model.ind_mapping[model.y[i]]
        for _ in 1:nSamples
            tot += log(logisticsoftmax(rand(p),class))/nSamples
        end
    end
    return tot
end
function ExpecLogLikelihood(model::SparseSoftMaxMultiClass)
    tot = 0.0
    nSamples = 200
    μ = model.κ.*model.μ;
    Σ = broadcast((κ,Σ,Ktilde)->[Ktilde[i] + dot(κ[i,:],Σ*κ[i,:]) for i in 1:model.nSamplesUsed],model.κ,model.Σ,model.Ktilde)

    for i in 1:model.nSamples
        p = MvNormal([μ[k][i] for k in 1:model.K],[sqrt(Σ[k][i]) for k in 1:model.K])
        class = model.ind_mapping[model.y[i]]
        for _ in 1:nSamples
            tot += log(softmax(rand(p),class))/nSamples
        end
    end
    return tot
end


function ExpecLogLikelihood(model::SparseLogisticSoftMaxMultiClass)
    tot = 0.0
    nSamples = 1000
    μ = model.κ.*model.μ;
    Σ = broadcast((κ,Σ,Ktilde)->[Ktilde[i] + dot(κ[i,:],Σ*κ[i,:]) for i in 1:model.nSamplesUsed],model.κ,model.Σ,model.Ktilde)

    for i in 1:model.nSamples
        p = MvNormal([μ[k][i] for k in 1:model.K],[sqrt(Σ[k][i]) for k in 1:model.K])
        class = model.ind_mapping[model.y[i]]
        for _ in 1:nSamples
            tot += log(logisticsoftmax(rand(p),class))/nSamples
        end
    end
    return tot
end

"Compute the variational updates for the full GP MultiClass"
function variational_updates!(model::Union{LogisticSoftMaxMultiClass{T},SoftMaxMultiClass{T},SparseSoftMaxMultiClass{T},SparseLogisticSoftMaxMultiClass{T}},iter::Integer) where T
    if iter == 1
        if typeof(model) <: Union{SparseSoftMaxMultiClass,SparseLogisticSoftMaxMultiClass}
            # model.L = [cholesky(model.Kmm[1]).L for _ in 1:model.K]
            # model.Σ .= copy.(model.Kmm)
        else
            model.L = [cholesky(model.Knn[1]).L for _ in 1:model.K]
            model.Σ .= copy.(model.Knn)
        end
        # println("Init check")
    end
    Gradient_Expec(model)

    # g_μ,g_L = compute_gradient_L(model,g_μ,g_Σ)
    # g_μ,g_Σ = compute_gradient_Σ(model,model.grad_μ,model.grad_Σ)
    g_μ,g_Λ = compute_gradient_Λ(model,model.grad_μ,model.grad_Σ)
    for k in 1:model.K
        updated = false; correct_coeff=1.0
        # up = update(model.Σ_optimizer[k],g_L)
        # up = update(model.Σ_optimizer[k],g_Σ[k])
        up = update(model.Σ_optimizer[k],g_η[k])
        while !updated
            try
                # @assert det(model.L[k]+correct_coeff*up) > 0
                # model.L[k] = LowerTriangular(model.L[k]+correct_coeff*up)
                # model.Σ[k] = Symmetric(model.L[k]*model.L[k]')

                # @assert isposdef(model.Σ[k]+correct_coeff*up)
                # model.Σ[k] = Symmetric(model.Σ[k]+correct_coeff*up)

                @assert isposdef(-(model.Λ[k] + correct_coeff*up))
                model.Λ[k] = model.Σ[k] +  correct_coeff*up
                model.Σ[k] = Symmetric(inv(model.Σ[k]))

                model.μ[k] .+= update(model.μ_optimizer[k],g_μ[k])
                updated = true
            catch
                model.Σ_optimizer[k].α *= 0.1
                correct_coeff *= 0.1
                println("Reducing value of α[$k], new value : $(model.Σ_optimizer[k].α)")
            end
        end
    end
    # display(det.(model.Σ))
    # display(isposdef.(model.Σ))
end
function compute_gradient_Σ(model::MultiClassGPModel,g_μ,g_Σ)
    grad_μ = [zero(model.μ[1]) for _ in 1:model.K]
    grad_Σ = [zero(model.Σ[1]) for _ in 1:model.K]
    for k in 1:model.K
        grad_μ[k] .= g_μ[k] - model.invK[k]*model.μ[k]
        grad_Σ[k] .= Diagonal(grad_Σ[k]) + Symmetric(0.5*(inv(model.Σ[k])-model.invK[k]))
    end
    return grad_μ,grad_Σ
end

function compute_gradient_Σ(model::Union{SparseSoftMaxMultiClass,SparseLogisticSoftMaxMultiClass},g_μ,g_Σ)
    grad_μ = [zero(model.μ[1]) for _ in 1:model.K]
    grad_Σ = [zero(model.Σ[1]) for _ in 1:model.K]
    for k in 1:model.K
        grad_μ[k] .= model.κ[k]'*g_μ[k] - model.invKmm[k]*model.μ[k]
        grad_Σ[k] .= Symmetric(model.κ[k]'*Diagonal(g_Σ[k])*model.κ[k] + 0.5*(inv(model.Σ[k])-model.invKmm[k]))
    end
    return grad_μ,grad_Σ
end

function compute_gradient_L(model::MultiClassGPModel,g_μ,g_Σ)
    grad_μ = [zero(model.μ[1]) for _ in 1:model.K]
    grad_Σ = [zero(model.L[1]) for _ in 1:model.K]
    for k in 1:model.K
        grad_L = zero(model.L[k])
        for i in 1:model.nFeatures
            for j in 1:i
                L_spec = Array(zero(model.L[k]))
                L_spec[j,:] = model.L[k][i,:]
                grad_L[i,j] = tr(Diagonal(g_Σ[k])*(L_spec+L_spec'))
            end
        end
        for i in 1:length(grad_μ[k])
            for j in 1:i
                L_spec = Array(zero(model.L[k]))
                L_spec[j,:] = model.L[k][i,:]
                grad_L[i,j] = tr(g_Σ[k]'*(L_spec+L_spec'))
            end
        end
        grad_μ[k] .= g_μ - model.invK[k]*model.μ[k]
        grad_L[k] .= grad_L + LowerTriangular(transpose(pinv(Array(model.L[k])))-model.L[k]*model.invK[k])
    end
    return grad_μ,grad_L
end

function compute_gradient_Λ(model::Union{SparseSoftMaxMultiClass,SparseLogisticSoftMaxMultiClass},g_μ,g_Σ)
    grad_μ = [zero(model.μ[1]) for _ in 1:model.K]
    grad_Σ = [zero(model.Σ[1]) for _ in 1:model.K]
    for k in 1:model.K
        grad_μ[k] .= model.Σ[k]*(model.κ[k]'*g_μ[k] - model.invKmm[k]*model.μ[k])
        grad_Σ[k] .= -2*Symmetric(model.κ[k]'*Diagonal(g_Σ[k])*model.κ[k] + 0.5*(model.Λ[k]-model.invKmm[k]))
    end
    return grad_μ,grad_Σ
end

###

function Gradient_Expec(model::SoftMaxMultiClass)
    nSamples = 200
    full_grad_μ = [zeros(model.nFeatures) for _ in 1:model.K]
    full_grad_Σ = [zeros(model.nFeatures) for _ in 1:model.K]
    for i in 1:model.nSamples
        p = MvNormal([model.μ[k][i] for k in 1:model.K],[sqrt(model.Σ[k][i,i]) for k in 1:model.K])
        grad_μ = zeros(model.K)
        grad_Σ = zeros(model.K)
        class = model.ind_mapping[model.y[i]]
        for _ in 1:nSamples
            samp = softmax(rand(p))
            s = samp[class]
            g_μ = grad_softmax(samp,class)
            grad_μ += g_μ./s
            grad_Σ += diag(hessian_softmax(samp,class))./s.-g_μ.^2 ./s^2
        end
        for k in 1:model.K
            full_grad_μ[k][i] = grad_μ[k]/nSamples
            full_grad_Σ[k][i] = 0.5*grad_Σ[k]/nSamples
        end
    end
    return full_grad_μ,full_grad_Σ
end

function Gradient_Expec(model::LogisticSoftMaxMultiClass)
    nSamples = 200
    full_grad_μ = [zeros(model.nFeatures) for _ in 1:model.K]
    full_grad_Σ = [zeros(model.nFeatures) for _ in 1:model.K]
    for i in 1:model.nSamples
        p = MvNormal([model.μ[k][i] for k in 1:model.K],[sqrt(model.Σ[k][i,i]) for k in 1:model.K])
        grad_μ = zeros(model.K)
        grad_Σ = zeros(model.K)
        class = model.ind_mapping[model.y[i]]
        for _ in 1:nSamples
            x = rand(p)
            samp = logisticsoftmax(x)
            σ = logit(x)
            s = samp[class]
            g_μ = grad_logisticsoftmax(samp,σ,class)
            grad_μ += g_μ./s
            grad_Σ += diag(hessian_logisticsoftmax(samp,σ,class))./s.-g_μ.^2 ./s^2
        end
        for k in 1:model.K
            full_grad_μ[k][i] = grad_μ[k]/nSamples
            full_grad_Σ[k][i] = 0.5*grad_Σ[k]/nSamples
        end
    end
    return full_grad_μ,full_grad_Σ
end

function Gradient_Expec(model::SparseSoftMaxMultiClass)
    nSamples = 100
    μ = model.κ.*model.μ; Σ = broadcast((κ,Σ,Ktilde)->[Ktilde[i] + dot(κ[i,:],Σ*κ[i,:]) for i in 1:model.nSamplesUsed],model.κ,model.Σ,model.Ktilde)
    # display
    for (iter,i) in enumerate(model.MBIndices)
        p = MvNormal([μ[k][iter] for k in 1:model.K],[sqrt(Σ[k][iter]) for k in 1:model.K])
        grad_μ = zeros(model.K)
        grad_Σ = zeros(model.K)
        class = model.ind_mapping[model.y[i]]
        for _ in 1:nSamples
            samp = softmax(rand(p))
            s = samp[class]
            g_μ = grad_softmax(samp,class)
            grad_μ += g_μ./s
            grad_Σ += diag(hessian_softmax(samp,class))./s.-g_μ.^2 ./s^2
        end
        for k in 1:model.K
            model.grad_μ[k][iter] = grad_μ[k]/nSamples
            model.grad_Σ[k][iter] = 0.5*grad_Σ[k]/nSamples
        end
    end
    return model.grad_μ,model.grad_Σ
end

function Gradient_Expec(model::SparseLogisticSoftMaxMultiClass)
    nSamples = 100
    μ = model.κ.*model.μ; Σ = broadcast((κ,Σ,Ktilde)->[Ktilde[i] + dot(κ[i,:],Σ*κ[i,:]) for i in 1:model.nSamplesUsed],model.κ,model.Σ,model.Ktilde)
    # display
    # vars = zeros(nSamples,model.nSamples*2)
    # av= (0.0,0.0);
    for (iter,i) in enumerate(model.MBIndices)
        p = MvNormal([μ[k][iter] for k in 1:model.K],[sqrt(Σ[k][iter]) for k in 1:model.K])
        grad_μ = zeros(model.K)
        grad_Σ = zeros(model.K)
        # M2 = zeros(2)
        class = model.ind_mapping[model.y[i]]
        for n in 1:nSamples
            x = rand(p)
            samp = logisticsoftmax(x)
            σ = logit(x)
            s = samp[class]
            g_μ = grad_logisticsoftmax(samp,σ,class)
            grad_μ += g_μ./s
            g_Σ = diag(hessian_logisticsoftmax(samp,σ,class))./s.-g_μ.^2 ./s^2
            grad_Σ += g_Σ
            # new_av = (grad_μ[1]/n,grad_Σ[1]/n)
            # if n==1
                # av = (grad_μ[1],grad_Σ[1])
            # end
            # M2 .+= [(g_μ[1]-av[1])*(g_μ[1]-new_av[1]),(g_Σ[1]-av[2])*(g_Σ[1]-new_av[2])]
            # vars[n,i] = M2[1]/(n)
            # vars[n,model.nSamples+i] = M2[2]/(n)
            # av = new_av
        end
        for k in 1:model.K
            model.grad_μ[k][iter] = grad_μ[k]/nSamples
            model.grad_Σ[k][iter] = 0.5*grad_Σ[k]/nSamples
        end
    end
    # push!(model.varMCMC,vars)
    return model.grad_μ,model.grad_Σ
end

"""Return KL Divergence for MvNormal for the MultiClass Model"""
function GaussianKL(model::MultiClassGPModel)
    if model.IndependentGPs
        return sum(0.5*(sum(model.invK[i].*(model.Σ[i]+model.μ[i]*transpose(model.μ[i])))-model.nSamples-logdet(model.Σ[i])-logdet(model.invK[i])) for i in model.KIndices)
    else
        return sum(0.5*(sum(model.invK[1].*(model.Σ[i]+model.μ[i]*transpose(model.μ[i])))-model.nSamples-logdet(model.Σ[i])-logdet(model.invK[1])) for i in model.KIndices)
    end
end


"""Return KL Divergence for MvNormal for the Sparse MultiClass Model"""
function GaussianKL(model::Union{SparseMultiClass,SparseSoftMaxMultiClass,SparseLogisticSoftMaxMultiClass})
    if model.IndependentGPs
        return sum(0.5*(sum(model.invKmm[i].*(model.Σ[i]+model.μ[i]*transpose(model.μ[i])))-model.m-logdet(model.Σ[i])+logdet(model.Kmm[i])) for i in model.KIndices)
    else
        return sum(0.5*(sum(model.invKmm[1].*(model.Σ[i]+model.μ[i]*transpose(model.μ[i])))-model.m-logdet(model.Σ[i])+logdet(model.Kmm[1])) for i in model.KIndices)
    end
end

function GammaImproperKL(model::GPModel)
    return sum(-model.α.+log(model.β[1]).-log.(gamma.(model.α)).-(1.0.-model.α).*digamma.(model.α))
end

function PoissonKL(model::MultiClass)
    return sum(γ->sum(γ.*(log.(γ).-1.0.-digamma.(model.α).+log.(model.β))+model.α./model.β),model.γ)
end

function PoissonKL(model::SparseMultiClass)
    return sum(γ->sum(γ.*(log.(γ).-1.0.-digamma.(model.α[model.MBIndices]).+log.(model.β))+model.α[model.MBIndices]./model.β),model.γ)
end

function PolyaGammaKL(model::MultiClass)
    return sum(broadcast((y,γ,c,θ)->sum(Array(y+γ).*log.(cosh.(0.5.*c))-0.5*(c.^2).*θ),model.Y[model.KIndices],model.γ,model.c,model.θ))
end


function PolyaGammaKL(model::SparseMultiClass)
    return sum(broadcast((y,γ,c,θ)->sum(Array(y[model.MBIndices]+γ).*log.(cosh.(0.5.*c))-0.5*(c.^2).*θ),model.Y[model.KIndices],model.γ,model.c,model.θ))
end

"""Return the gradient of the ELBO given the kernel hyperparameters"""
function hyperparameter_gradient_function(model::MultiClassGPModel{T}) where T
    if model.IndependentGPs
        A = [(model.invK[i]*(model.Σ[i]+model.µ[i]*model.μ[i]')-I)*model.invK[i] for i in model.KIndices]
        return (function(J,Kindex,index)
                    return 0.5*sum(J.*transpose(A[index]))
                end,
                function(kernel,Kindex,index)
                    return 0.5/getvariance(kernel)*sum(model.Knn[i].*A[index]')
                end)
    else
        A = [(model.invK[1]*(model.Σ[i]+model.µ[i]*model.μ[i]')-I)*model.invK[i] for i in model.KIndices]
        V = Matrix{T}(undef,model.nSamples,model.nSamples)
        return (function(J,Kindex,index)
            return 0.5*model.KStochCoeff*sum([sum(J.*transpose(A[i])) for i in 1:model.nClassesUsed])
                end,
                function(kernel)
                    return 0.5/getvariance(kernel)*sum(tr.(A))
                end)
    end
end

"""Return the gradient of the ELBO given the kernel hyperparameters"""
function hyperparameter_gradient_function(model::SparseMultiClass{T}) where T
    #General values used for all gradients
     C2 = broadcast((μ::Vector{T},Σ::Symmetric{T,Matrix{T}})->μ*transpose(μ) + Σ,model.μ[model.KIndices],model.Σ[model.KIndices])
    #preallocation
    ι = Matrix{T}(undef,model.nSamplesUsed,model.m)
    Jtilde = Vector{T}(undef,model.nSamplesUsed)
    V = Matrix{T}(undef,model.m,model.m)
    κθ = model.κ'.*Diagonal.(model.θ)
    if model.IndependentGPs
        A = Matrix{T}(undef,model.m,model.m)
        return (function(Jmm::LinearAlgebra.Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},Kindex::Int64,index::Int64) where {T}
                    # ι = (Jnm-model.κ[index]*Jmm)*model.invKmm[Kindex]
                    mul!(ι,(Jnm-model.κ[index]*Jmm),model.invKmm[Kindex])
                    # Jtilde = Jnn - sum(ι.*model.Knm[index],dims=2)[:] - sum(model.κ[index].*Jnm,dims=2)[:]
                    Jnn .+= - sum(ι.*model.Knm[index],dims=2)[:] - sum(model.κ[index].*Jnm,dims=2)[:]
                    # V = Matrix(model.invKmm[Kindex])*Matrix(Jmm)
                    mul!(V,model.invKmm[Kindex],Matrix(Jmm))
                    trV = tr(V)
                    V *= model.invKmm[Kindex]
                    A = add_transpose!(κθ[index]*ι)
                    grad = sum(V.*C2[index])
                    grad += - model.StochCoeff*sum(A.*C2[index])
                    grad += - trV
                    grad += - model.StochCoeff*dot(model.θ[index],Jnn)
                    grad += model.StochCoeff * dot(Array(model.Y[Kindex][model.MBIndices]) - model.γ[index], ι*model.μ[Kindex])
                    grad *= 0.5
                    return grad
                    # return 0.5*(sum((V*model.invKmm[Kindex]).*F2[index])-model.StochCoeff*sum(A.*F2[index])
                    #         - trV
                    #         - model.StochCoeff*dot(model.θ[index],Jtilde)
                    #         + model.StochCoeff * dot(Array(model.Y[Kindex][model.MBIndices]) - model.γ[index], ι*model.μ[Kindex]))
         end, #end of function(Js)
                function(kernel::KernelModule.Kernel,Kindex::Int64,index::Int64)
                    return 0.5/(getvariance(kernel))*(sum(model.invKmm[Kindex].*C2[index])-model.StochCoeff * dot(model.θ[index],model.Ktilde[index])-model.m)
                end)
    else
        A = [Matrix{T}(undef,model.m,model.m) for _ in 1:model.nClassesUsed]
        return (function(Jmm::LinearAlgebra.Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},Kindex::Int64,index::Int64) where {T}
                    mul!((Jnm-model.κ[1]*Jmm),model.invKmm[1])
                    Jnn .+= - sum(ι.*model.Knm[1],dims=2)[:] - sum(model.κ[1].*Jnm,dims=2)[:]
                    A .= broadcast(kc::Matrix{T}->add_transpose!(kc*ι),κθ)
                    mul!(V,Matrix(model.invKmm[1]),Matrix(Jmm))
                    TraceV = -tr(V)
                    V*= model.invKmm[1]
                    return 0.5*(model.KStochCoeff*sum(broadcast((c2,a,θ,y,γ,μ)->(sum(V.*c2)-sum(a.*c2)*model.StochCoeff)-
                        model.StochCoeff*dot(θ,Jtilde)+
                        model.StochCoeff*dot(Array(y[model.MBIndices])-γ,ι*μ),
                        C2,A,model.θ,model.Y[model.KIndices],model.γ,model.μ[model.KIndices]))+model.K*TraceV)
        end,#end of function(Js)
        function(kernel::KernelModule.Kernel)
            # println(mean(F2[index]))
            return 0.5/(getvariance(kernel))*sum(broadcast((c2,θ)->sum(model.invKmm[1].*c2)-model.StochCoeff * dot(θ,model.Ktilde[1])-model.m,C2,model.θ))
        end)
       end
end

"""Return the gradient of the ELBO given the kernel hyperparameters"""
function hyperparameter_gradient_function(model::Union{SparseSoftMaxMultiClass{T},SparseLogisticSoftMaxMultiClass{T}}) where T
    if model.IndependentGPs
        A = [(model.invKmm[i]*(model.Σ[i]+model.µ[i]*model.μ[i]')-I)*model.invKmm[i] for i in model.KIndices]
        ι = Matrix{T}(undef,model.nSamplesUsed,model.m)
        # Jtilde = Vector{T}(undef,model.nSamplesUsed)
        return (function(Jmm::LinearAlgebra.Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},Kindex::Int64,index::Int64) where {T}
                    mul!(ι,(Jnm-model.κ[index]*Jmm),model.invKmm[Kindex])
                    Jnn .+= - sum(ι.*model.Knm[index],dims=2)[:] - sum(model.κ[index].*Jnm,dims=2)[:]
                    dμ = dot(model.grad_μ[Kindex],ι*model.μ[Kindex])
                    dΣ = dot(model.grad_Σ[Kindex],Jnn+2.0*sum((ι*model.Σ[Kindex]).*model.κ[index],dims=2)[:])
                    return model.StochCoeff*(dμ+dΣ)+ 0.5*sum(Jmm.*transpose(A[index]))
                end,
                function(kernel,Kindex,index)
                    return 0.5/getvariance(kernel)*(2.0*model.StochCoeff*dot(model.grad_Σ[Kindex],model.Ktilde[index])+tr(model.Kmm[index]*A[index]))
                end)
    else
        A = [model.invK[1]*(model.Σ[i]+model.µ[i]*model.μ[i]')-Diagonal{T}(I,model.nSamples) for i in model.KIndices]
        V = Matrix{T}(undef,model.nSamples,model.nSamples)
        return (function(J,Kindex,index)
            V = model.invK[1]*J #invK*J
            return 0.5*model.KStochCoeff*sum([sum(V.*transpose(A[i])) for i in 1:model.nClassesUsed])
                end,
                function(kernel)
                    return 0.5/getvariance(kernel)*sum(tr.(A))
                end)
    end
end


"""Return a function computing the gradient of the ELBO given the inducing point locations"""
function inducingpoints_gradient(model::SparseMultiClass{T}) where T
    if model.IndependentGPs
        gradients_inducing_points = zero(model.inducingPoints[1])
        C2 = broadcast((μ::Vector{T},Σ::Symmetric{T,Matrix{T}})->μ*transpose(μ) + Σ,model.μ[model.KIndices],model.Σ[model.KIndices])
        #preallocation
        ι = Matrix{T}(undef,model.nSamplesUsed,model.m)
        Jtilde = Vector{T}(undef,model.nSamplesUsed)
        V = Matrix{T}(undef,model.m,model.m)
        κθ = model.κ'.*Diagonal.(model.θ)
        for (ic,c) in enumerate(model.KIndices)
            for i in 1:model.m #Iterate over the points
                Jnm,Jmm = computeIndPointsJ(model,i) #TODO
                for j in 1:model.nDim #iterate over the dimensions
                    mul!(ι,(Jnm[j,:,:]-model.κ[ic]*Jmm[j,:,:]),model.invKmm[c])
                    Jtilde = - sum(ι.*model.Knm[i],dims=2)[:] - sum(model.κ[index].*Jnm,dims=2)[:]
                    # V = Matrix(model.invKmm[Kindex])*Matrix(Jmm)
                    mul!(V,model.invKmm[Kindex],Matrix(Jmm))
                    trV = tr(V)
                    V *= model.invKmm[Kindex]
                    A = add_transpose!(κθ[index]*ι)
                    grad = sum(V.*C2[index])
                    grad += - model.StochCoeff*sum(A.*C2[index])
                    grad += - trV
                    grad += - model.StochCoeff*dot(model.θ[index],Jnn)
                    grad += model.StochCoeff * dot(Array(model.Y[Kindex][model.MBIndices]) - model.γ[index], ι*model.μ[Kindex])
                    grad *= 0.5
                    gradients_inducing_points[c][i,j] = grad
                end
            end
        end
        return gradients_inducing_points
    else
        gradients_inducing_points = zero(model.inducingPoints[1])

    end
end
