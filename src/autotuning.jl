function kernelderivative(kernel::Kernel{T,<:ScaleTransform},X) where {T}
    p = get_params(kernel)
    J = reshape(ForwardDiff.jacobian(x->kernelmatrix(typeof(kernel)(x),X,obsdim=1),p),size(X,1),size(X,1),length(p))
    return [J[:,:,i] for i in 1:length(p)]
end

function kernelderivative(kernel::Kernel{T,<:ScaleTransform},X,Y) where {T}
    p = get_params(kernel)
    J = reshape(ForwardDiff.jacobian(x->kernelmatrix(typeof(kernel)(x),X,Y,obsdim=1),p),size(X,1),size(Y,1),length(p))
    return [J[:,:,i] for i in 1:length(p)]
end

function kerneldiagderivative(kernel::Kernel{T,<:ScaleTransform},X)
    p = get_params(kernel)
    J = reshape(ForwardDiff.jacobian(x->kerneldiagmatrix(typeof(kernel)(x),X,obsdim=1),p),size(X,1),length(p))
    return [J[:,i] for i in 1:length(p)]
end

function update_hyperparameters!(model::VGP)
    update_hyperparameters!.(model.f,model.X,model.opt_ρ,model.opt_σ,model.opt_μ₀)
end

function update_hyperparameters!(model::SVGP)
    update_hyperparameters!.(model.f,model.inference.X,model.opt_ρ,model.opt_σ,model.opt_μ₀,model.likelihood,[get_y(model)],model.inference)
end

## Update all hyperparameters for the full batch GP models ##
function update_hyperparameters!(gp::_VGP{T},X::AbstractMatrix,opt_ρ::Optimizer,opt_σ_k::Optimizer,opt_μ₀::Optimizer) where {T}
    Jnn = kernelderivative(gp.kernel,X)
    f_l,f_v,f_μ₀ = hyperparameter_gradient_function(gp)
    grads_ρ = compute_hyperparameter_gradient(gp.kernel,f_l,Jnn)
    grads_σ_k = f_v(gp.kernel,gp.σ_k)
    grads_μ₀ = f_μ₀()

    apply_gradients_lengthscale!(opt_ρ,gp.kernel,grads_ρ) #Send the derivative of the matrix to the specific gradient of the model
    apply_gradients_variance!(opt_σ_k,gp,grads_σ_k) #Send the derivative of the matrix to the specific gradient of the model
    apply_gradients_mean_prior!(opt_μ₀,gp.μ₀,grads_μ₀)

    model.inference.HyperParametersUpdated = true
end

## Update all hyperparameters for the sparse variational GP models ##
function update_hyperparameters!(model::_SVGP{T},,opt_ρ::Optimizer,opt_σ_k::Optimizer,opt_μ₀::Optimizer,l::Likelihood,y::AbstractVector) where {T}
    Jmm = kernelderivative(gp.kernel,gp.Z)
    Jnm = kernelderivative(gp.kernel,X,gp.Z)
    Jnn = kerneldiagderivative(gp.kernel,X)
    f_ρ,f_σ_k,f_μ₀ = hyperparameter_gradient_function(gp)
    grads_l = compute_hyperparameter_gradient(gp.kernel,f_l,Jmm,Jnm,Jnn,l,y)
    grads_v = f_v(gp.kernel,gp.σ_k)
    grads_μ₀ = f_μ₀()
    if !isnothing(gp.Z.opt)
        Z_gradients = inducingpoints_gradient(model) #Compute the gradient given the inducing points location
        gp.Z.Z += GradDescent.update(gp.Z.opt,Z_gradients) #Apply the gradients on the location
    end
    apply_gradients_lengthscale!(opt_ρ,gp.kernel,grads_l)
    apply_gradients_variance!(opt_σ_k,gp.kernel,grads_v)
    apply_gradients_mean_prior!(opt_μ₀,gp.μ₀,grads_μ₀)
    model.inference.HyperParametersUpdated = true
end


## Return the derivative of the KL divergence between the posterior and the GP prior ##
function hyperparameter_KL_gradient(J::AbstractMatrix{T},A::AbstractMatrix{T}) where {T<:Real}
    return 0.5*opt_trace(J,A)
end


## Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model ##
function hyperparameter_gradient_function(gp::_VGP{T}) where {T<:Real}
    A = ([Diagonal{T}(I,gp.dim)]-gp.K\(gp.Σ+(gp.µ-gp.μ₀)*transpose(gp.μ-gp.μ₀)))/gp.K
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
    return (function(Jmm,Jnm,Jnn,l,y,i)
                return (hyperparameter_expec_gradient(gp,ι,κΣ,Jmm,Jnm,Jnn,l,y,i)-hyperparameter_KL_gradient(Jmm,A))
            end,
            function(kernel::Kernel,σ_k::Real,l,y,i)
                return one(T)/σ_k*(
                        - i.ρ*dot(∇E_Σ(l,y),gp.K̃)
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
function hyperparameter_expec_gradient(gp::_SVGP{T},ι::Matrix{T},κΣ::Matrix{T},Jmm::Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},l::Likelihood,i::AnalyticVI,y::AbstractVector) where {T<:Real}
    ι .= (Jnm-gp.κ*Jmm)/gp.K
    Jnn .-= opt_diag(ι,gp.Knm) + opt_diag(gp.κ,Jnm)
    dμ = dot(∇E_μ(l,y),ι*gp.μ)
    dΣ = -dot(∇E_Σ(l,y),Jnn)
    dΣ += -dot(∇E_Σ(l,y),2.0*(opt_diag(ι,κΣ)))
    dΣ += -dot(∇E_Σ(l,y),2.0*(ι*gp.μ).*(gp.κ*gp.μ))
    return i.ρ*(dμ+dΣ)
end

## Gradient with respect to hyperparameter with shared priors for numerical VI ##
function hyperparameter_expec_gradient(model::_SVGP{T},ι::Matrix{T},κΣ::Vector{Matrix{T}},Jmm::Matrix{T},Jnm::Matrix{T},Jnn::Vector{T},l::Likelihood,i::NumericalVI,y::AbstractVector) where {T<:Real}
    ι .= (Jnm-gp.κ*Jmm)/gp.K
    Jnn .-= opt_diag(ι,gp.Knm) + opt_diag(gp.κ,Jnm)
    dμ = dot(∇E_μ(i),ι*gp.μ)
    dΣ = dot(∇E_Σ(i),Jnn+2.0*opt_diag(ι,κΣ))
    return i.ρ*(dμ+dΣ)
end


## Return a function computing the gradient of the ELBO given the inducing point locations ##
function inducingpoints_gradient(model::SVGP{T}) where {T<:Real}
    if model.IndependentPriors
        gradients_inducing_points = zero(model.Z[1])
        A = ([I].-model.invKmm.*(model.Σ.+model.µ.*transpose.(model.μ))).*model.invKmm
        #preallocation
        ι = Matrix{T}(undef,model.nSamplesUsed,model.nInducingPoints)
        for k in 1:model.nPrior
            for i in 1:model.nInducingPoints #Iterate over the points
                Jnm,Jmm = computeIndPointsJ(model,i) #TODO
                for j in 1:model.nDim #iterate over the dimensions
                    @views mul!(ι,(Jnm[j,:,:]-model.κ[k]*Jmm[j,:,:]),model.invKmm[k])
                    @views gradients_inducing_points[c][i,j] =  hyperparameter_expec_gradient(model,ι,κΣ,Jmm[j,:,:],Jnm[j,:,:],zeros(T,model.nSamplesUsed))-hyperparameter_KL_gradient(Jmm[j,:,:],A[k])
                end
            end
        end
        return gradients_inducing_points
    else
        @warn "Inducing points for shared prior not implemented yet"
        gradients_inducing_points = zero(model.inducingPoints[1]) #TODO
    end
end

## Apply gradients on mean prior ##
function apply_gradients_mean_prior!(μ::AbstractVector{<:Real},grad_μ::AbstractVector{<:Real},opt::Optimizer)
    μ .+= update(opt,grad_μ)
end
