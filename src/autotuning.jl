include("autotuning_utils.jl")
function update_hyperparameters!(model::VGP)
    update_hyperparameters!.(model.f,[model.X])
    model.inference.HyperParametersUpdated = true
end

function update_hyperparameters!(model::SVGP)
    update_hyperparameters!.(model.f,[model.inference.x],model.likelihood,model.inference,[get_y(model)])
    model.inference.HyperParametersUpdated = true
end

## Update all hyperparameters for the full batch GP models ##
function update_hyperparameters!(gp::_VGP{T},X) where {T}
    if !isnothing(gp.opt_ρ) || !isnothing(gp.opt_σ) || !isnothing(get_opt(gp.μ₀))
        f_l,f_v,f_μ₀ = hyperparameter_gradient_function(gp)
        if !isnothing(gp.opt_ρ)
            Jnn = kernelderivative(gp.kernel,X)
            grads_ρ = compute_hyperparameter_gradient(gp.kernel,f_l,Jnn)
        end
        if !isnothing(gp.opt_σ)
            grads_σ_k = f_v(gp.kernel,gp.σ_k)
        end
        if !isnothing(get_opt(gp.μ₀))
            grads_μ₀ = f_μ₀()
        end

        if !isnothing(gp.opt_ρ)
            apply_gradients_lengthscale!(gp.opt_ρ,gp.kernel,grads_ρ) #Send the derivative of the matrix to the specific gradient of the model
        end
        if !isnothing(gp.opt_σ)
            apply_gradients_variance!(gp,grads_σ_k) #Send the derivative of the matrix to the specific gradient of the model
        end
        if !isnothing(get_opt(gp.μ₀))
            apply_gradients_mean_prior!(gp.μ₀,grads_μ₀)
        end
    end
end

## Update all hyperparameters for the sparse variational GP models ##
function update_hyperparameters!(gp::_SVGP{T},X,l::Likelihood,i::Inference,y::AbstractVector) where {T}
    if !isnothing(gp.opt_ρ) || !isnothing(gp.opt_σ) || !isnothing(get_opt(gp.μ₀))
        f_ρ,f_σ_k,f_μ₀ = hyperparameter_gradient_function(gp)
        if !isnothing(gp.opt_ρ)
            Jmm = kernelderivative(gp.kernel,gp.Z.Z)
            Jnm = kernelderivative(gp.kernel,X,gp.Z.Z)
            Jnn = kerneldiagderivative(gp.kernel,X)
            grads_l = compute_hyperparameter_gradient(gp.kernel,f_ρ,Jmm,Jnm,Jnn,l,i,y)
        end
        if !isnothing(gp.opt_σ)
            grads_σ_k = f_σ_k(gp.kernel,gp.σ_k,l,i,y)
        end
        if !isnothing(get_opt(gp.μ₀))
            grads_μ₀ = f_μ₀()
        end
        if !isnothing(gp.Z.opt)
            Z_gradients = inducingpoints_gradient(gp,X,l,i,y) #Compute the gradient given the inducing points location
            gp.Z.Z .+= GradDescent.update(gp.Z.opt,Z_gradients) #Apply the gradients on the location
        end
        if !isnothing(gp.opt_ρ)
            apply_gradients_lengthscale!(gp.opt_ρ,gp.kernel,grads_l)
        end
        if !isnothing(gp.opt_σ)
            apply_gradients_variance!(gp,grads_σ_k)
        end
        if !isnothing(get_opt(gp.μ₀))
            apply_gradients_mean_prior!(gp.μ₀,grads_μ₀)
        end
    end
end


## Return the derivative of the KL divergence between the posterior and the GP prior ##
function hyperparameter_KL_gradient(J::AbstractMatrix{T},A::AbstractMatrix{T}) where {T<:Real}
    return 0.5*opt_trace(J,A)
end


## Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model ##
function hyperparameter_gradient_function(gp::_VGP{T}) where {T<:Real}
    A = (Diagonal{T}(I,gp.dim)-gp.K\(gp.Σ+(gp.µ-gp.μ₀)*transpose(gp.μ-gp.μ₀)))/gp.K
    return (function(Jnn)
                return -hyperparameter_KL_gradient(Jnn,A)
            end,
            function(kernel,σ_k)
                return -one(T)/σ_k*hyperparameter_KL_gradient(gp.K,A)
            end,
            function()
                return -gp.K\(gp.μ₀-gp.μ)
            end)
end

function hyperparameter_local_gradient_function(model::VGP{T}) where {T<:Real}
    A = ([Diagonal{T}(I,model.nFeatures)].-model.likelihood.invK.*(model.likelihood.Σ.+(model.likelihood.µ.-model.likelihood.μ₀).*transpose.(model.likelihood.μ.-model.likelihood.μ₀))).*model.likelihood.invK
    return (function(Jnn,index)
                return -hyperparameter_KL_gradient(Jnn,A[index])
            end,
            function(kernel,index)
                return -1.0/getvariance(kernel)*hyperparameter_KL_gradient(model.Knn[index],A[index])
            end,
            function(index)
                return -model.likelihood.invK[index]*(model.likelihood.μ₀[index]-model.likelihood.μ[index])
            end)
end

## Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model ##
function hyperparameter_gradient_function(gp::_SVGP{T}) where {T<:Real}
    A = (Diagonal{T}(I,gp.dim).-gp.K\(gp.Σ.+(gp.µ-gp.μ₀)*transpose(gp.μ-gp.μ₀)))/gp.K
    ι = similar(gp.κ) #Empty container to save data allocation
    κΣ = gp.κ*gp.Σ
    return (function(Jmm,Jnm,Jnn,l,i,y)
                return (hyperparameter_expec_gradient(gp,l,i,y,ι,κΣ,Jmm,Jnm,Jnn)-hyperparameter_KL_gradient(Jmm,A))
            end,
            function(kernel::Kernel,σ_k::Real,l,i,y)
                return one(T)/σ_k*(
                        - i.ρ*dot(∇E_Σ(l,i,y),gp.K̃)
                        - hyperparameter_KL_gradient(gp.K,A))
            end,
            function()
                return -gp.K\(gp.μ₀-gp.μ)
            end)
end

function hyperparameter_gradient_function(model::VStP{T}) where {T<:Real}
    A = ([Diagonal{T}(I,model.nFeatures)].-invK(model).*(model.Σ.+(model.µ.-model.μ₀).*transpose.(model.μ.-model.μ₀))).*invK(model)
    if model.IndependentPriors
        return (function(Jnn,index)
                    return -hyperparameter_KL_gradient(Jnn,A[index])
                end,
                function(kernel,index)
                    return -1.0/getvariance(kernel)*hyperparameter_KL_gradient(model.Knn[index].*model.χ[index],A[index])
                end,
                function(index)
                    return -invK(model,index)*(model.μ₀[index]-model.μ[index])
                end)
    else
        return (function(Jnn,index)
            return -sum(hyperparameter_KL_gradient.([Jnn],A))
                end,
                function(kernel,index)
                    return -1.0/getvariance(kernel)*sum(hyperparameter_KL_gradient.(model.Knn[index].*model.χ[index],A) for index in 1:model.nPrior)
                end,
                function(index)
                    return -sum(invK(model).*(model.μ₀.-model.μ))
                end)
    end
end

## Gradient with respect to hyperparameter with independent priors for analytical VI ##
function hyperparameter_expec_gradient(gp::_SVGP{T},l::Likelihood,i::AnalyticVI,y::AbstractVector,ι::AbstractMatrix{T},κΣ::AbstractMatrix{T},Jmm::AbstractMatrix{T},Jnm::AbstractMatrix{T},Jnn::AbstractVector{T}) where {T<:Real}
    ι .= (Jnm-gp.κ*Jmm)/gp.K
    Jnn .-= opt_diag(ι,gp.Knm) + opt_diag(gp.κ,Jnm)
    dμ = dot(∇E_μ(l,i,y),ι*gp.μ)
    dΣ = -dot(∇E_Σ(l,i,y),Jnn)
    dΣ += -dot(∇E_Σ(l,i,y),2.0*(opt_diag(ι,κΣ)))
    dΣ += -dot(∇E_Σ(l,i,y),2.0*(ι*gp.μ).*(gp.κ*gp.μ))
    return i.ρ*(dμ+dΣ)
end

## Gradient with respect to hyperparameter with shared priors for numerical VI ##
function hyperparameter_expec_gradient(gp::_SVGP{T},l::Likelihood,i::NumericalVI,y::AbstractVector,ι::AbstractMatrix{T},κΣ::AbstractMatrix{T},Jmm::AbstractMatrix{T},Jnm::AbstractMatrix{T},Jnn::AbstractVector{T}) where {T<:Real}
    ι .= (Jnm-gp.κ*Jmm)/gp.K
    Jnn .-= opt_diag(ι,gp.Knm) + opt_diag(gp.κ,Jnm)
    dμ = dot(∇E_μ(l,i,y),ι*gp.μ)
    dΣ = dot(∇E_Σ(l,i,y),Jnn+2.0*opt_diag(ι,κΣ))
    return i.ρ*(dμ+dΣ)
end


## Return a function computing the gradient of the ELBO given the inducing point locations ##
function inducingpoints_gradient(gp::_SVGP{T},X,l::Likelihood,i::Inference,y::AbstractVector) where {T<:Real}
    gradient_inducing_points = similar(gp.Z.Z)
    A = (I-gp.K\(gp.Σ+gp.µ*transpose(gp.μ)))/gp.K
    #preallocation
    ι = similar(gp.κ)
    Jmm,Jnm = indpoint_derivative(gp.kernel,gp.Z),indpoint_derivative(gp.kernel,X,gp.Z)
    κΣ = gp.κ*gp.Σ
    for j in 1:gp.dim #Iterate over the points
        for k in 1:size(gp.Z,2) #iterate over the dimensions
            @views ι = (Jnm[:,:,j,k]-gp.κ*Jmm[:,:,j,k])/gp.K
            @views gradient_inducing_points[j,k] = hyperparameter_expec_gradient(gp,l,i,y,ι,κΣ,Jmm[:,:,j,k],Jnm[:,:,j,k],zero(gp.K̃))-hyperparameter_KL_gradient(Jmm[:,:,j,k],A)
        end
    end
    return gradient_inducing_points
end
