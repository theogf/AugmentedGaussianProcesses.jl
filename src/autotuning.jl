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
        end
    end
    return gradient_inducing_points
end
