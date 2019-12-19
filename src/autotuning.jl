include("zygote_rules.jl")
include("autotuning_utils.jl")

function update_hyperparameters!(model::Union{GP,VGP})
    update_hyperparameters!.(model.f,get_Z(model))
    model.inference.HyperParametersUpdated = true
end

function update_hyperparameters!(model::SVGP)
    update_hyperparameters!.(model.f,[model.inference.xview],∇E_μ(model.likelihood,model.inference.vi_opt[1],get_y(model)),∇E_Σ(model.likelihood,model.inference.vi_opt[1],get_y(model)),model.inference,model.inference.vi_opt)
    model.inference.HyperParametersUpdated = true
end

function update_hyperparameters!(model::MOSVGP)
    update_hyperparameters!.(model.f,[model.inference.xview],∇E_μ(model),∇E_Σ(model),model.inference,model.inference.vi_opt)
    model.inference.HyperParametersUpdated = true
end

## Update all hyperparameters for the full batch GP models ##
function update_hyperparameters!(gp::Union{_GP{T},_VGP{T}},X::AbstractMatrix) where {T}
    if !isnothing(gp.opt)
        f_l,f_v,f_μ₀ = hyperparameter_gradient_function(gp,X)
        grads = ∇L_ρ(f_l,gp,X)
        grads.grads[gp.σ_k] = f_v(first(gp.σ_k))
        grads.grads[gp.μ₀] = f_μ₀()

        apply_grads_kernel_params!(gp.opt,gp.kernel,grads) # Apply gradients to the kernel parameters
        apply_grads_kernel_variance!(gp.opt,gp,grads[gp.σ_k]) #Send the derivative of the matrix to the specific gradient of the model
        apply_gradients_mean_prior!(gp.opt,gp.μ₀,grads[gp.μ₀],X)
    end
end

## Update all hyperparameters for the sparse variational GP models ##
function update_hyperparameters!(gp::_SVGP{T},X,∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::AbstractOptimizer) where {T}
    if !isnothing(gp.opt)
        f_ρ,f_σ_k,f_μ₀ = hyperparameter_gradient_function(gp)
        grads =  ∇L_ρ(f_ρ,gp,X,∇E_μ,∇E_Σ,i,opt)
        grads.grads[gp.σ_k] = f_σ_k(first(gp.σ_k),∇E_Σ,i,opt)
        grads.grads[gp.μ₀] = f_μ₀()
    end
    if !isnothing(gp.Z.opt)
        Z_gradients = inducingpoints_gradient(gp,X,∇E_μ,∇E_Σ,i,opt) #Compute the gradient given the inducing points location
        gp.Z.Z .+= Flux.apply!(gp.Z.opt,gp.Z.Z,Z_gradients) #Apply the gradients on the location
    end
    if !isnothing(gp.opt)
        apply_grads_kernel_params!(gp.opt,gp.kernel,grads) # Apply gradients to the kernel parameters
        apply_grads_kernel_variance!(gp.opt,gp,grads[gp.σ_k]) #Send the derivative of the matrix to the specific gradient of the model
        apply_gradients_mean_prior!(gp.opt,gp.μ₀,grads[gp.μ₀],X)
    end
end


## Return the derivative of the KL divergence between the posterior and the GP prior ##
function hyperparameter_KL_gradient(J::AbstractMatrix{T},A::AbstractMatrix{T}) where {T<:Real}
    return 0.5*opt_trace(J,A)
end


function hyperparameter_gradient_function(gp::_GP{T},X::AbstractMatrix) where {T}
    μ₀ = gp.μ₀(X)
    A = (inv(gp.K).mat-gp.μ*transpose(gp.μ))
    return (function(Jnn)
                return -hyperparameter_KL_gradient(Jnn,A)
            end,
            function(σ_k)
                return -one(T)/σ_k*hyperparameter_KL_gradient(gp.K.mat,A)
            end,
            function()
                return -gp.μ
            end)
end

## Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model ##
function hyperparameter_gradient_function(gp::_VGP{T},X::AbstractMatrix) where {T<:Real}
    μ₀ = gp.μ₀(X)
    A = (I-gp.K\(gp.Σ+(gp.µ-gp.μ₀(X))*transpose(gp.μ-μ₀)))/gp.K.mat
    return (function(Jnn)
                return -hyperparameter_KL_gradient(Jnn,A)
            end,
            function(σ_k)
                return -one(T)/σ_k*hyperparameter_KL_gradient(gp.K.mat,A)
            end,
            function()
                return -gp.K.mat\(μ₀-gp.μ)
            end)
end

## Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse latent GP ##
function hyperparameter_gradient_function(gp::_SVGP{T}) where {T<:Real}
    μ₀ = gp.μ₀(gp.Z.Z)
    A = (Diagonal{T}(I,gp.dim).-gp.K\(gp.Σ.+(gp.µ-μ₀)*transpose(gp.μ-μ₀)))/gp.K.mat
    κΣ = gp.κ*gp.Σ
    return (function(Jmm,Jnm,Jnn,∇E_μ,∇E_Σ,i,opt)
                return (hyperparameter_expec_gradient(gp,∇E_μ,∇E_Σ,i,opt,κΣ,Jmm,Jnm,Jnn)-hyperparameter_KL_gradient(Jmm,A))
            end,
            function(σ_k::Real,∇E_Σ,i,opt)
                return one(T)/σ_k*(
                        - i.ρ*dot(∇E_Σ,gp.K̃)
                        - hyperparameter_KL_gradient(gp.K.mat,A))
            end,
            function()
                return -gp.K.mat\(μ₀-gp.μ)
            end)
end

function hyperparameter_gradient_function(model::VStP{T},X::AbstractMatrix) where {T<:Real}
    μ₀ = gp.μ₀(X)
    A = (Diagonal{T}(I,gp.dim).-gp.K\(gp.Σ.+(gp.µ-μ₀)*transpose(gp.μ-μ₀)))/gp.K.mat
    return (function(Jnn)
                return -hyperparameter_KL_gradient(Jnn,A)
            end,
            function(σ_k::Real)
                return -one(T)/σ_k*hyperparameter_KL_gradient(gp.K.mat*gp.χ,A)
            end,
            function()
                return -gp.K.mat\(μ₀-gp.μ)
            end)
function hyperparameter_gradient_function(model::OnlineVGP{<:Likelihood,<:Inference,T}) where {T<:Real}
"""Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model"""
    A = ([Diagonal{T}(I,model.nFeatures)].-model.invKmm.*(model.Σ.+model.µ.*transpose.(model.μ))).*model.invKmm
    ι = Matrix{T}(undef,model.inference.nSamplesUsed,model.nFeatures) #Empty container to save data allocation
    global ιₐ = Matrix{T}(undef,size(model.Zₐ[1],1),size(model.Z[1],1)) #Empty container to save data allocation
    κΣ = model.κ.*model.Σ
                    return (hyperparameter_expec_gradient(model,ι,κΣ[index],Jmm,Jnm,Jnn,index)
    κₐΣ = model.κₐ.*model.Σ
        return (function(Jmm::Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},Jab::Matrix{T},Jaa::Symmetric{T,Matrix{T}},index::Int)
    if model.IndependentPriors
                     +hyperparameter_online_gradient(model,ιₐ,κₐΣ[index],Jmm,Jab,Jaa,index)
                     - hyperparameter_KL_gradient(Jmm,A[index]))
                end,
                function(kernel::Kernel{T},index::Int)
                    return 1.0/getvariance(kernel)*(
                            - dot(model.likelihood.θ[index],model.K̃[index])
                            - 0.5*opt_trace(model.invDₐ[index],model.K̃ₐ[index])
                            - hyperparameter_KL_gradient(model.Kmm[index],A[index]))
                function(index)
                    return -model.invKmm[index]*(model.μ₀[index]-model.μ[index])
                end,
                function(model)
                end,
                        inducingpoints_gradient(model,A,ι,ιₐ,κΣ,κₐΣ)
                end)
    else
        return (function(Jmm::Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},Jab::Matrix{T},Jaa::Symmetric{T,Matrix{T}},index::Int)
                    return  (hyperparameter_expec_gradient(model,ι,κΣ,Jmm,Jnm,Jnn)
                end,
                         + hyperparameter_online_gradient(model,ιₐ,κₐΣ,Jmm,Jab,Jaa,index)  - sum(hyperparameter_KL_gradient.([Jmm],A)))
                function(kernel::Kernel{T},index::Int)
                    return 1.0/getvariance(kernel)*(sum(
                            -dot(model.likelihood.θ[i],model.K̃[1]) for i in 1:model.nLatent)
                            - 0.5*sum(opt_trace.(model.invDₐ,model.K̃ₐ))
                            - sum(hyperparameter_KL_gradient.(model.Kmm,A)))
                end,
                function(index)
                    return -sum(model.invKmm.*(model.μ₀.-model.μ))
                end,
                function(model)
                end)
                        inducingpoints_gradient(model,A,ι,ιₐ,κΣ,κₐΣ)
    end
end
end

## Gradient with respect to hyperparameter for analytical VI ##
function hyperparameter_expec_gradient(gp::_SVGP{T},∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::AnalyticVI,opt::AVIOptimizer,κΣ::AbstractMatrix{T},Jmm::AbstractMatrix{T},Jnm::AbstractMatrix{T},Jnn::AbstractVector{T}) where {T<:Real}
    ι = (Jnm-gp.κ*Jmm)/gp.K.mat
    Jnn = Jnn - (opt_diag(ι,gp.Knm) + opt_diag(gp.κ,Jnm))
    dμ = dot(∇E_μ,ι*gp.μ)
    dΣ = -dot(∇E_Σ,Jnn)
    dΣ += -dot(∇E_Σ,2.0*(opt_diag(ι,κΣ)))
    dΣ += -dot(∇E_Σ,2.0*(ι*gp.μ).*(gp.κ*gp.μ))
    return i.ρ*(dμ+dΣ)
end


## Gradient with respect to hyperparameters for numerical VI ##
function hyperparameter_expec_gradient(gp::_SVGP{T},∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::NumericalVI,opt::NVIOptimizer,ι::AbstractMatrix{T},κΣ::AbstractMatrix{T},Jmm::AbstractMatrix{T},Jnm::AbstractMatrix{T},Jnn::AbstractVector{T}) where {T<:Real}
    ι .= (Jnm-gp.κ*Jmm)/gp.K.mat
    Jnn .-= opt_diag(ι,gp.Knm) + opt_diag(gp.κ,Jnm)
    dμ = dot(∇E_μ,ι*gp.μ)
    dΣ = dot(∇E_Σ,Jnn+2.0*opt_diag(ι,κΣ))
    return i.ρ*(dμ+dΣ)
function hyperparameter_online_gradient(model::OnlineVGP{<:Likelihood{T},<:Inference{T},T},ιₐ::Matrix{T},κₐΣ::Matrix{T},Jmm::Symmetric{T,Matrix{T}},Jab::Matrix{T},Jaa::Symmetric{T,Matrix{T}},index::Integer) where {T<:Real}
    mul!(ιₐ,(Jab-model.κₐ[index]*Jmm),model.invKmm[index])
    # trace_term = sum(opt_trace.([model.invDₐ[index]],[Jaa,2*ιₐ*transpose(κₐΣ),-(2*Jab+model.κ[index]*Jmm)*model.invKmm[index]*transpose(model.Kab[index])]))
    trace_term = sum(opt_trace.([model.invDₐ[index]],[Jaa,2*ιₐ*transpose(κₐΣ),-ιₐ*transpose(model.Kab[index]),- model.κₐ[index]*transpose(Jab)]))
    term_1 = -2.0*dot(model.prevη₁[index],ιₐ*model.μ[index])
    return -0.5*(trace_term+term_1+term_2)
    term_2 = 2.0*dot(ιₐ*model.μ[index],model.invDₐ[index]*model.κₐ[index]*model.μ[index])
end

function hyperparameter_online_gradient(model::OnlineVGP{<:Likelihood{T},<:Inference{T},T},ιₐ::Matrix{T},κₐΣ::Vector{Matrix{T}},Jmm::Symmetric{T,Matrix{T}},Jab::Matrix{T},Jaa::Symmetric{T,Matrix{T}},index::Integer) where {T<:Real}
    mul!(ιₐ,(Jab-model.κₐ[1]*Jmm),model.invKmm[1])
    J_q = Jaa - (ιₐ*transpose(model.Kab[1]) + model.κₐ[1]*transpose(Jab))
    trace_term = sum(sum(opt_trace.([model.invDₐ[j]],[J_q,2*ιₐ*transpose(κₐΣ[j])])) for j in 1:model.nLatent)
    term_1 = sum(-2.0*dot(model.prevη₁[j],ιₐ*model.μ[j]) for j in 1:model.nLatent)
    term_2 = sum(2.0*dot(ιₐ*model.μ[j],model.invDₐ[j]*model.κₐ[1]*model.μ[j]) for j in 1:model.nLatent)
    return -0.5*(trace_term+term_1+term_2)
end

## Return a function computing the gradient of the ELBO given the inducing point locations ##
function inducingpoints_gradient(gp::_SVGP{T},X,∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::AbstractOptimizer) where {T<:Real}
    gradient_inducing_points = similar(gp.Z.Z)
    A = (I-gp.K\(gp.Σ+gp.µ*transpose(gp.μ)))/gp.K
    #preallocation
    ι = similar(gp.κ)
    Jmm,Jnm = indpoint_derivative(gp.kernel,gp.Z),indpoint_derivative(gp.kernel,X,gp.Z)
    κΣ = gp.κ*gp.Σ
    for j in 1:gp.dim #Iterate over the points
        for k in 1:size(gp.Z,2) #iterate over the dimensions
            @views ι = (Jnm[:,:,j,k]-gp.κ*Jmm[:,:,j,k])/gp.K
            @views gradient_inducing_points[j,k] = hyperparameter_expec_gradient(gp,∇E_μ,∇E_Σ,i,opt,ι,κΣ,Jmm[:,:,j,k],Jnm[:,:,j,k],zero(gp.K̃))-hyperparameter_KL_gradient(Jmm[:,:,j,k],A)
function inducingpoints_gradient(model::OnlineVGP{<:Likelihood{T},<:Inference{T},T},A,ι,ιₐ,κΣ,κₐΣ) where {T<:Real}
        gradients_inducing_points = [zeros(T,model.nFeatures,model.nDim) for _ in 1:model.nLatent]
        for k in 1:model.nPrior
            for i in 1:model.nFeatures #Iterate over the points
                Jnm,Jab,Jmm = computeIndPointsJ(model,i,k) #TODO
                for j in 1:model.nDim #iterate over the dimensions
                    mul!(ι,(Jnm[j,:,:]-model.κ[k]*Jmm[j,:,:]),model.invKmm[k])
                    gradients_inducing_points[k][i,j] =  (hyperparameter_expec_gradient(model,ι,κΣ[k],Symmetric(Jmm[j,:,:]),Jnm[j,:,:],zeros(T,model.inference.nSamplesUsed),k)
                    + hyperparameter_online_gradient(model,ιₐ,κₐΣ[k],Symmetric(Jmm[j,:,:]),Jab[j,:,:],Symmetric(zeros(T,size(model.Zₐ[k],1),size(model.Zₐ[k],1))),k)
                    - hyperparameter_KL_gradient(Jmm[j,:,:],A[k]))
                end
            end
"""Return a function computing the gradient of the ELBO given the inducing point locations"""
    if model.IndependentPriors
        end
    end
    return gradient_inducing_points
end
