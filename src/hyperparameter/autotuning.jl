include("autotuning_utils.jl")
include("zygote_rules.jl")
include("forwarddiff_rules.jl")

function update_hyperparameters!(model::Union{GP,VGP})
    update_hyperparameters!.(model.f,get_Z(model))
    model.inference.HyperParametersUpdated = true
end

function update_hyperparameters!(model::Union{SVGP,OnlineSVGP})
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
        aduse = K_ADBACKEND[] == :auto ? ADBACKEND[] : K_ADBACKEND[]
        global grads = if aduse == :forward_diff
            ∇L_ρ_forward(f_l,gp,X)
        elseif aduse == :reverse_diff
            ∇L_ρ_reverse(f_l,gp,X)
        end
        grads[gp.σ_k] = f_v(first(gp.σ_k))
        grads[gp.μ₀] = f_μ₀()
        apply_grads_kernel_params!(gp.opt,gp.kernel,grads) # Apply gradients to the kernel parameters
        # apply_grads_kernel_variance!(gp.opt,gp,grads[gp.σ_k]) #Send the derivative of the matrix to the specific gradient of the model
        apply_gradients_mean_prior!(gp.opt,gp.μ₀,grads[gp.μ₀],X)
    end
end

## Update all hyperparameters for the sparse variational GP models ##
function update_hyperparameters!(gp::Union{_SVGP{T},_OSVGP{T}},X,∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::Inference,opt::AbstractOptimizer) where {T}
    if !isnothing(gp.opt)
        f_ρ,f_Z,f_σ_k,f_μ₀ = hyperparameter_gradient_function(gp)
        k_aduse = K_ADBACKEND[] == :auto ? ADBACKEND[] : K_ADBACKEND[]
        global grads = if k_aduse == :forward_diff
            ∇L_ρ_forward(f_ρ,gp,X,∇E_μ,∇E_Σ,i,opt)
        elseif k_aduse == :reverse_diff
            ∇L_ρ_reverse(f_ρ,gp,X,∇E_μ,∇E_Σ,i,opt)
        end
        # @show grads[gp.kernel.transform.s]
        grads[gp.σ_k] = f_σ_k(first(gp.σ_k),∇E_Σ,i,opt)
        grads[gp.μ₀] = f_μ₀()
    end
    if !isnothing(gp.Z.opt)
        Z_aduse = Z_ADBACKEND[] == :auto ? ADBACKEND[] : Z_ADBACKEND[]
        global Z_gradients = if Z_aduse == :forward_diff
               Z_gradient_forward(gp,f_Z,X,∇E_μ,∇E_Σ,i,opt) #Compute the gradient given the inducing points location
           elseif Z_aduse == :reverse_diff
               Z_gradient_reverse(gp,f_Z,X,∇E_μ,∇E_Σ,i,opt)
           end
        gp.Z.Z .+= Flux.Optimise.apply!(gp.Z.opt,gp.Z.Z,Z_gradients) #Apply the gradients on the location
    end
    if !isnothing(gp.opt)
        apply_grads_kernel_params!(gp.opt,gp.kernel,grads) # Apply gradients to the kernel parameters
        apply_grads_kernel_variance!(gp.opt,gp,grads[gp.σ_k]) # Apply gradient on the kernel variance
        apply_gradients_mean_prior!(gp.opt,gp.μ₀,grads[gp.μ₀],X)
    end
end


## Return the derivative of the KL divergence between the posterior and the GP prior ##
function hyperparameter_KL_gradient(J::AbstractMatrix{T},A::AbstractMatrix{T}) where {T<:Real}
    return 0.5*opt_trace(J,A)
end


function hyperparameter_gradient_function(gp::_GP{T},X::AbstractMatrix) where {T}
    A = (inv(gp.K).mat-(gp.μ)*transpose(gp.μ))# μ = inv(K+σ²)*(y-μ₀)
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
    A = (I-gp.K\(gp.Σ+(gp.µ-gp.μ₀(X))*transpose(gp.μ-μ₀)))/gp.K
    return (function(Jnn)
                return -hyperparameter_KL_gradient(Jnn,A)
            end,
            function(σ_k)
                return -one(T)/σ_k*hyperparameter_KL_gradient(gp.K.mat,A)
            end,
            function()
                return gp.K\(gp.μ-μ₀)
            end)
end

## Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse latent GP ##
function hyperparameter_gradient_function(gp::_SVGP{T}) where {T<:Real}
    μ₀ = gp.μ₀(gp.Z.Z)
    A = (Diagonal{T}(I,gp.dim)-gp.K\(gp.Σ+(gp.µ-μ₀)*transpose(gp.μ-μ₀)))/gp.K
    κΣ = gp.κ*gp.Σ
    return (function(Jmm,Jnm,Jnn,∇E_μ,∇E_Σ,i,opt)
                return (hyperparameter_expec_gradient(gp,∇E_μ,∇E_Σ,i,opt,κΣ,Jmm,Jnm,Jnn)-hyperparameter_KL_gradient(Jmm,A))
            end,
            function(Jmm,Jnm,∇E_μ,∇E_Σ,i,opt)
                hyperparameter_expec_gradient(gp,∇E_μ,∇E_Σ,i,opt,ι,κΣ,Jmm,Jnm,zero(gp.K̃))-hyperparameter_KL_gradient(Jmm,A)
            end,
            function(σ_k::Real,∇E_Σ,i,opt)
                return one(T)/σ_k*(
                        - i.ρ*dot(∇E_Σ,gp.K̃)
                        - hyperparameter_KL_gradient(gp.K.mat,A))
            end,
            function()
                return gp.K\(gp.μ-μ₀)
            end)
end

function hyperparameter_gradient_function(model::VStP{T},X::AbstractMatrix) where {T<:Real}
    μ₀ = gp.μ₀(X)
    A = (Diagonal{T}(I,gp.dim).-gp.K\(gp.Σ.+(gp.µ-μ₀)*transpose(gp.μ-μ₀)))/gp.K
    return (function(Jnn)
                return -hyperparameter_KL_gradient(Jnn,A)
            end,
            function(σ_k::Real)
                return -one(T)/σ_k*hyperparameter_KL_gradient(gp.K.mat*gp.χ,A)
            end,
            function()
                return -(gp.K\(μ₀-gp.μ))
            end)
end


function hyperparameter_gradient_function(gp::_OSVGP{T}) where {T<:Real}
    μ₀ = gp.μ₀(gp.Z.Z)
    A = (Diagonal{T}(I,gp.dim)-gp.K\(gp.Σ+(gp.µ-μ₀)*transpose(gp.μ-μ₀)))/gp.K
    κΣ = gp.κ*gp.Σ
    κₐΣ = gp.κₐ*gp.Σ
    return (function(Jmm,Jnm,Jnn,Jab,Jaa,∇E_μ,∇E_Σ,i,opt)
                ∇E = hyperparameter_expec_gradient(gp,∇E_μ,∇E_Σ,i,opt,κΣ,Jmm,Jnm,Jnn)
                ∇KLₐ = hyperparameter_online_gradient(gp,κₐΣ,Jmm,Jab,Jaa)
                ∇KL =  hyperparameter_KL_gradient(Jmm,A)
                # return ∇E - ∇KL
                return ∇E + ∇KLₐ - ∇KL
                end,
                function(Jmm,Jnm,Jab,∇E_μ,∇E_Σ,i,opt)
                    hyperparameter_expec_gradient(gp,∇E_μ,∇E_Σ,i,opt,κΣ,Jmm,Jnm,zero(gp.K̃))+ hyperparameter_online_gradient(gp,κₐΣ,Jmm,Jab,zeros(T,size(gp.Zₐ,1),size(gp.Zₐ,1)))-hyperparameter_KL_gradient(Jmm,A)
                end,
                function(σ_k::Real,∇E_Σ,i,opt)
                    return one(T)/σ_k*(
                                - i.ρ*dot(∇E_Σ,gp.K̃)
                                - 0.5*opt_trace(gp.invDₐ,gp.K̃ₐ)
                                - hyperparameter_KL_gradient(gp.K.mat,A))
                end,
                function()
                    return -(gp.K\(μ₀-gp.μ))
                end)
end

## Gradient with respect to hyperparameter for analytical VI ##
function hyperparameter_expec_gradient(gp::Union{_SVGP{T},_OSVGP{T}},∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::AnalyticVI,opt::AVIOptimizer,κΣ::AbstractMatrix{<:Real},Jmm::AbstractMatrix{<:Real},Jnm::AbstractMatrix{<:Real},Jnn::AbstractVector{<:Real}) where {T<:Real}
    ι = (Jnm-gp.κ*Jmm)/gp.K
    J̃ = Jnn - (opt_diag(ι,gp.Knm) + opt_diag(gp.κ,Jnm))
    dμ = dot(∇E_μ,ι*gp.μ)
    dΣ = -dot(∇E_Σ,J̃)
    dΣ += -dot(∇E_Σ,2.0*(opt_diag(ι,κΣ)))
    dΣ += -dot(∇E_Σ,2.0*(ι*gp.μ).*(gp.κ*gp.μ))
    return i.ρ*(dμ+dΣ)
end


## Gradient with respect to hyperparameters for numerical VI ##
function hyperparameter_expec_gradient(gp::_SVGP{T},∇E_μ::AbstractVector{T},∇E_Σ::AbstractVector{T},i::NumericalVI,opt::NVIOptimizer,ι::AbstractMatrix{T},κΣ::AbstractMatrix{T},Jmm::AbstractMatrix{<:Real},Jnm::AbstractMatrix{<:Real},Jnn::AbstractVector{<:Real}) where {T<:Real}
    ι .= (Jnm-gp.κ*Jmm)/gp.K
    Jnn .-= opt_diag(ι,gp.Knm) + opt_diag(gp.κ,Jnm)
    dμ = dot(∇E_μ,ι*gp.μ)
    dΣ = dot(∇E_Σ,Jnn+2.0*opt_diag(ι,κΣ))
    return i.ρ*(dμ+dΣ)
end

function hyperparameter_online_gradient(gp::_OSVGP{T},κₐΣ::Matrix{T},Jmm::AbstractMatrix,Jab::AbstractMatrix{T},Jaa::AbstractMatrix{T}) where {T<:Real}
    ιₐ = (Jab-gp.κₐ*Jmm)/gp.K
    trace_term = -0.5*sum(opt_trace.([gp.invDₐ],[Jaa,2*ιₐ*transpose(κₐΣ),-ιₐ*transpose(gp.Kab),-gp.κₐ*transpose(Jab)]))
    term_1 = dot(gp.prevη₁,ιₐ*gp.μ)
    term_2 = -dot(ιₐ*gp.μ,gp.invDₐ*gp.κₐ*gp.μ)
    return trace_term+term_1+term_2
end
