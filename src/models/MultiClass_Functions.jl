#Set of functions for the multiclass model

"Update the local variational parameters of the full batch GP Multiclass"
function local_update!(model::MultiClass)
    model.c .= broadcast((Σ,μ)->sqrt.(Σ.+μ.^2),diag.(model.Σ[model.KIndices]),model.μ[model.KIndices])
    for _ in 1:2
        model.γ .= broadcast((c,μ)->0.5./(model.β.*cosh.(0.5.*c)).*exp.(digamma.(model.α).-0.5.*μ),model.c,model.μ[model.KIndices])
        model.α .= [1+model.KStochCoeff*sum([γ[i] for γ in model.γ]) for i in 1:model.nSamples]
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
function local_update!(model::SparseMultiClass)
    model.c = broadcast((μ::Vector{Float64},Σ::Symmetric{Float64,Matrix{Float64}},κ::Matrix{Float64},ktilde::Vector{Float64})->sqrt.(ktilde+vec(sum((κ*Σ).*κ,dims=2))+(κ*μ).^2),model.μ[model.KIndices],model.Σ[model.KIndices],model.κ,model.Ktilde)
    for _ in 1:model.nInnerLoops
        model.γ .= broadcast((c,κ,μ)->0.5./(model.β.*cosh.(0.5.*c)).*exp.(digamma.(model.α).-0.5.*κ*μ),model.c,model.κ,model.μ[model.KIndices])
        model.α[model.MBIndices] .= [1+model.KStochCoeff*sum([γ[i] for γ in model.γ]) for i in 1:model.nSamplesUsed]
    end
    model.θ .= broadcast((y,γ::Vector{Float64},c::Vector{Float64})->0.5.*Array(γ+y[model.MBIndices])./c.*tanh.(0.5.*c),model.Y[model.KIndices],model.γ,model.c)
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
function global_update!(model::SparseMultiClass,grad_1::Vector{Vector{Float64}},grad_2::Vector{Matrix{Float64}})
    model.η_1[model.KIndices] .= (1.0.-model.ρ_s[model.KIndices]).*model.η_1[model.KIndices] + model.ρ_s[model.KIndices].*grad_1;
    model.η_2[model.KIndices] .= Symmetric.(model.η_2[model.KIndices].*(1.0.-model.ρ_s[model.KIndices]) + model.ρ_s[model.KIndices].*grad_2) #Update of the natural parameters with noisy/full natural gradient
    #TODO Temporary fix until LinearAlgebra has corrected it
    model.Σ[model.KIndices] .= -inv.(model.η_2[model.KIndices]).*0.5; model.μ[model.KIndices] .= model.Σ[model.KIndices].*model.η_1[model.KIndices] #Back to the distribution parameters (needed for α updates)
end

"""Compute the natural gradient of the ELBO given the natural parameters"""
function natural_gradient!(model::MultiClass)
        model.η_1[model.KIndices] .= broadcast((y,γ)->0.5*Array(y-γ),model.Y[model.KIndices],model.γ)
        if model.IndependentGPs
            model.η_2[model.KIndices] .= broadcast((y,θ,invK)->Symmetric(-0.5*(Diagonal(θ)+invK)),model.Y[model.KIndices],model.θ,model.invK[model.KIndices])
        else
            model.η_2[model.KIndices] .= broadcast((y,θ)->Symmetric(-0.5*(Diagonal(θ)+model.invK[1])),model.Y[model.KIndices],model.θ[2:end])
        end
end

"""Compute the natural gradient of the ELBO given the natural parameters"""
function natural_gradient(model::SparseMultiClass)
    grad_1 = broadcast((y::SparseVector{Int64,Int64},κ::Matrix{Float64},γ::Vector{Float64})->0.5*model.StochCoeff*κ'*Array(y[model.MBIndices]-γ),model.Y[model.KIndices],model.κ,model.γ)
    grad_2 = broadcast((y::SparseVector{Int64,Int64},κ::Matrix{Float64},θ::Vector{Float64},invKmm::Symmetric{Float64,Matrix{Float64}})->-0.5*(model.StochCoeff*κ'*Diagonal(θ)*κ+invKmm),model.Y[model.KIndices],model.κ,model.θ,model.invKmm[model.IndependentGPs ? model.KIndices : :])
    return grad_1, grad_2
end


"""Return the negative ELBO for the MultiClass model"""
function ELBO(model::MultiClass)
    model.c .= broadcast((Σ,μ)->sqrt.(Σ.+μ.^2),diag.(model.Σ[model.KIndices]),model.μ[model.KIndices])
    ELBO_v = 0.0
    ELBO_v += ExpecLogLikelihood(model)
    ELBO_v += -model.KStochCoeff*GaussianKL(model)
    ELBO_v += -GammaImproperKL(model)
    ELBO_v += -model.KStochCoeff*PoissonKL(model)
    ELBO_v += -model.KStochCoeff*PolyaGammaKL(model)
    return -ELBO_v
end

"""Return the negative ELBO for the Sparse MultiClass model"""
function ELBO(model::SparseMultiClass)
    ELBO_v = 0.0
    ELBO_v += model.StochCoeff*ExpecLogLikelihood(model)
    ELBO_v += -model.KStochCoeff*GaussianKL(model)
    ELBO_v += -GammaImproperKL(model)
    ELBO_v += -model.KStochCoeff*model.StochCoeff*PoissonKL(model)
    ELBO_v += -model.KStochCoeff*model.StochCoeff*PolyaGammaKL(model)
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

"""Return KL Divergence for MvNormal for the MultiClass Model"""
function GaussianKL(model::MultiClass)
    if model.IndependentGPs
        return sum(0.5*(sum(model.invK[i].*(model.Σ[i]+model.μ[i]*transpose(model.μ[i])))-model.nSamples-logdet(model.Σ[i])-logdet(model.invK[i])) for i in model.KIndices)
    else
        return        sum(0.5*(sum(model.invK[1].*(model.Σ[i]+model.μ[i]*transpose(model.μ[i])))-model.nSamples-logdet(model.Σ[i])-logdet(model.invK[1])) for i in model.KIndices)
    end
end

"""Return KL Divergence for MvNormal for the Sparse MultiClass Model"""
function GaussianKL(model::SparseMultiClass)
    if model.IndependentGPs
        return sum(0.5*(sum(model.invKmm[i].*(model.Σ[i]+model.μ[i]*transpose(model.μ[i])))-model.m-logdet(model.Σ[i])-logdet(model.invKmm[i])) for i in model.KIndices)
    else
        return        sum(0.5*(sum(model.invKmm[1].*(model.Σ[i]+model.μ[i]*transpose(model.μ[i])))-model.m-logdet(model.Σ[i])-logdet(model.invKmm[1])) for i in model.KIndices)
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
    C2 = broadcast((μ::Vector{Float64},Σ::Symmetric{Float64,Matrix{Float64}})->μ*transpose(μ) + Σ,model.μ[model.KIndices],model.Σ[model.KIndices])
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
                    grad = sum(V.*C2[index])
                    grad += - model.StochCoeff*sum(A.*C2[index])
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
                    return 0.5/(getvariance(kernel))*(sum(model.invKmm[Kindex].*C2[index])-model.StochCoeff * dot(C[index],model.Ktilde[index])-model.m)
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
                    return 0.5*(model.KStochCoeff*sum(broadcast((c2,a,c,y,γ,μ)->                (sum(V.*c2)-sum(a.*c2)*model.StochCoeff)
                        -model.StochCoeff*dot(c,Jtilde)
                        + model.StochCoeff*dot(Array(y[model.MBIndices])-γ,ι*μ),
                        C2,A,C,model.Y[model.KIndices],model.γ,model.μ[model.KIndices]))+model.K*TraceV)
        end,#end of function(Js)
        function(kernel::Kernel)
            # println(mean(F2[index]))
            return 0.5/(getvariance(kernel))*sum(broadcast((c2,c)->sum(model.invKmm[1].*c2)-model.StochCoeff * dot(c,model.Ktilde[1])-model.m,C2,C))
        end)
       end
end
