"""Update all hyperparameters for the full batch GP models"""
function  update_hyperparameters!(model::Union{VGP,GP})
    Jnn = kernelderivativematrix.([model.X],model.kernel)
    f_l,f_v = hyperparameter_gradient_function(model)
    grads_l = map(compute_hyperparameter_gradient,model.kernel,fill(f_l,model.nPrior),Jnn,1:model.nPrior)
    grads_v = map(f_v,model.kernel,1:model.nPrior)
    apply_gradients_lengthscale!.(model.kernel,grads_l) #Send the derivative of the matrix to the specific gradient of the model
    apply_gradients_variance!.(model.kernel,grads_v) #Send the derivative of the matrix to the specific gradient of the model
    model.inference.HyperParametersUpdated = true
end

"""Update all hyperparameters for the full batch GP models"""
function update_hyperparameters!(model::SVGP{<:Likelihood,<:Inference,T}) where {T<:Real}
    Jmm = kernelderivativematrix.(model.Z,model.kernel)
    Jnm = kernelderivativematrix.([model.X[model.inference.MBIndices,:]],model.Z,model.kernel)
    Jnn = kernelderivativediagmatrix.([model.X[model.inference.MBIndices,:]],model.kernel)
    f_l,f_v = hyperparameter_gradient_function(model)
    grads_l = map(compute_hyperparameter_gradient,model.kernel,fill(f_l,model.nPrior),Jmm,Jnm,Jnn,collect(1:model.nPrior))
    grads_v = map(f_v,model.kernel,1:model.nPrior)
    if model.OptimizeInducingPoints
        Z_gradients = inducingpoints_gradient(model) #Compute the gradient given the inducing points location
        model.Z += GradDescent.update(model.optimizer,Z_gradients) #Apply the gradients on the location
    end
    apply_gradients_lengthscale!.(model.kernel,grads_l)
    apply_gradients_variance!.(model.kernel,grads_v)
    model.inference.HyperParametersUpdated = true
end

"""Update all hyperparameters for the full batch GP models"""
function update_hyperparameters!(model::OnlineVGP{<:Likelihood,<:Inference,T}) where {T<:Real}
    global Jmm = kernelderivativematrix.(model.Z,model.kernel)
    Jnm = kernelderivativematrix.([model.X],model.Z,model.kernel)
    Jnn = kernelderivativediagmatrix.([model.X],model.kernel)
    global Jab = kernelderivativematrix.(model.Zₐ,model.Z,model.kernel)
    global Jaa = kernelderivativematrix.(model.Zₐ,model.kernel)
    f_l,f_v = hyperparameter_gradient_function(model)
    grads_l = map(compute_hyperparameter_gradient,model.kernel,fill(f_l,model.nPrior),Jmm,Jnm,Jnn,Jab,Jaa,collect(1:model.nPrior))
    grads_v = map(f_v,model.kernel,1:model.nPrior)
    if model.OptimizeInducingPoints
        Z_gradients = inducingpoints_gradient(model) #Compute the gradient given the inducing points location
        model.Z += GradDescent.update(model.optimizer,Z_gradients) #Apply the gradients on the location
    end
    apply_gradients_lengthscale!.(model.kernel,grads_l)
    apply_gradients_variance!.(model.kernel,grads_v)
    model.inference.HyperParametersUpdated = true
end

"""Return the derivative of the KL divergence between the posterior and the GP prior"""
function hyperparameter_KL_gradient(J::AbstractMatrix{T},A::AbstractMatrix{T}) where {T<:Real}
    return 0.5*opt_trace(J,A)
end


"""Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model"""
function hyperparameter_gradient_function(model::VGP) where {T<:Real}
    A = ([Diagonal{T}(I,model.nFeature)].-model.invKnn.*(model.Σ.+model.µ.*transpose.(model.μ))).*model.invKnn
    if model.IndependentPriors
        return (function(Jnn,index)
                    return -hyperparameter_KL_gradient(Jnn,A[index])
                end,
                function(kernel,index)
                    return -1.0/getvariance(kernel)*hyperparameter_KL_gradient(model.Knn[index],A[index])
                end)
    else
        return (function(Jnn,index)
            return -sum(hyperparameter_KL_gradient.([Jnn],A))
                end,
                function(kernel,index)
                    return -1.0/getvariance(kernel)*sum(hyperparameter_KL_gradient.(model.Knn,A))
                end)
    end
end



"""Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model"""
function hyperparameter_gradient_function(model::SVGP{<:Likelihood,<:Inference,T}) where {T<:Real}
    A = ([Diagonal{T}(I,model.nFeature)].-model.invKmm.*(model.Σ.+model.µ.*transpose.(model.μ))).*model.invKmm
    ι = Matrix{T}(undef,model.inference.nSamplesUsed,model.nFeature) #Empty container to save data allocation
    κΣ = model.κ.*model.Σ
    if model.IndependentPriors
        return (function(Jmm::Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},index::Int)
                    return (hyperparameter_expec_gradient(model,ι,κΣ[index],Jmm,Jnm,Jnn,index)-hyperparameter_KL_gradient(Jmm,A[index]))
                end,
                function(kernel::Kernel{T},index::Int)
                    return 1.0/getvariance(kernel)*(
                            - model.inference.ρ*dot(expec_Σ(model,index),model.K̃[index])
                            - hyperparameter_KL_gradient(model.Kmm[index],A[index]))
                end)
    else
        return (function(Jmm::Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},index::Int)
                    return  (hyperparameter_expec_gradient(model,ι,κΣ,Jmm,Jnm,Jnn)
                           - sum(hyperparameter_KL_gradient.([Jmm],A)))
                end,
                function(kernel::Kernel{T},index::Int)
                    return 1.0/getvariance(kernel)*(model.inference.ρ*sum(
                            -dot(expec_Σ(model,i),model.K̃[1]) for i in 1:model.nLatent)
                            - sum(hyperparameter_KL_gradient.(model.Kmm,A)))
                end)
    end
end

"""Return functions computing gradients of the ELBO given the kernel hyperparameters for a non-sparse model"""
function hyperparameter_gradient_function(model::OnlineVGP{<:Likelihood,<:Inference,T}) where {T<:Real}
    A = ([Diagonal{T}(I,model.nFeature)].-model.invKmm.*(model.Σ.+model.µ.*transpose.(model.μ))).*model.invKmm
    ι = Matrix{T}(undef,model.inference.nSamplesUsed,model.nFeature) #Empty container to save data allocation
    global ιₐ = Matrix{T}(undef,size(model.Zₐ[1],1),size(model.Z[1],1)) #Empty container to save data allocation
    κΣ = model.κ.*model.Σ
    κₐΣ = model.κₐ.*model.Σ
    if model.IndependentPriors
        return (function(Jmm::Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},Jab::Matrix{T},Jaa::Symmetric{T,Matrix{T}},index::Int)
                    return (hyperparameter_expec_gradient(model,ι,κΣ[index],Jmm,Jnm,Jnn,index) + hyperparameter_online_gradient(model,ιₐ,κₐΣ[index],Jmm,Jab,Jaa,index) - hyperparameter_KL_gradient(Jmm,A[index]))
                end,
                function(kernel::Kernel{T},index::Int)
                    return 1.0/getvariance(kernel)*(
                            - dot(expec_Σ(model,index),model.K̃[index])
                            - 0.5*opt_trace(model.invDₐ[index],model.K̃ₐ[index])
                            - hyperparameter_KL_gradient(model.Kmm[index],A[index]))
                end)
    else
        return (function(Jmm::Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},Jab::Matrix{T},Jaa::Symmetric{T,Matrix{T}},index::Int)
                    return  (hyperparameter_expec_gradient(model,ι,κΣ,Jmm,Jnm,Jnn)
                         + hyperparameter_online_gradient(model,ιₐ,κₐΣ,Jmm,Jab,Jaa,index)  - sum(hyperparameter_KL_gradient.([Jmm],A)))
                end,
                function(kernel::Kernel{T},index::Int)
                    return 1.0/getvariance(kernel)*(sum(
                            -dot(expec_Σ(model,i),model.K̃[1]) for i in 1:model.nLatent)
                            - 0.5*sum(opt_trace.(model.invDₐ,model.K̃ₐ))
                            - sum(hyperparameter_KL_gradient.(model.Kmm,A)))
                end)
    end
end


function hyperparameter_expec_gradient(model::SparseGP{<:Likelihood{T},<:Inference{T},T},ι::Matrix{T},κΣ::Matrix{T},Jmm::Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T},index::Integer) where {T<:Real}
    mul!(ι,(Jnm-model.κ[index]*Jmm),model.invKmm[index])
    Jnn .-= opt_diag(ι,model.Knm[index]) + opt_diag(model.κ[index],Jnm)
    dμ = dot(expec_μ(model,index),ι*model.μ[index])
    dΣ = -dot(expec_Σ(model,index),Jnn+2.0*(opt_diag(ι*model.Σ[index],model.κ[index])))
    if model.inference isa AnalyticVI
        dΣ += -dot(expec_Σ(model,index),2.0*(ι*model.μ[index]).*(model.κ[index]*model.μ[index]))
    end
    return model.inference.ρ*(dμ+dΣ)
end

function hyperparameter_expec_gradient(model::SparseGP{<:Likelihood{T},<:Inference{T},T},ι::Matrix{T},κΣ::Vector{Matrix{T}},Jmm::Symmetric{T,Matrix{T}},Jnm::Matrix{T},Jnn::Vector{T}) where {T<:Real}
    mul!(ι,(Jnm-model.κ[1]*Jmm),model.invKmm[1])
    Jnn .-= opt_diag(ι,model.Knm[1]) + opt_diag(model.κ[1],Jnm)
    dμ = sum(dot(expec_μ(model,i),ι*model.μ[i]) for i in 1:model.nLatent)
    dΣ = -sum(dot(expec_Σ(model,i),Jnn+2.0*opt_diag(ι,κΣ[i])) for i in 1:model.nLatent)
    if model.inference isa AnalyticVI
        dΣ += -sum(dot(expec_Σ(model,i),2.0*(ι*model.μ[i]).*(model.κ[1]*model.μ[i])) for i in 1:model.nLatent)
    end
    return model.inference.ρ*(dμ+dΣ)
end

function hyperparameter_online_gradient(model::OnlineVGP{<:Likelihood{T},<:Inference{T},T},ιₐ::Matrix{T},κₐΣ::Matrix{T},Jmm::Symmetric{T,Matrix{T}},Jab::Matrix{T},Jaa::Symmetric{T,Matrix{T}},index::Integer) where {T<:Real}
    mul!(ιₐ,(Jab-model.κₐ[index]*Jmm),model.invKmm[index])
    # trace_term = sum(opt_trace.([model.invDₐ[index]],[Jaa,2*ιₐ*transpose(κₐΣ),-(2*Jab+model.κ[index]*Jmm)*model.invKmm[index]*transpose(model.Kab[index])]))
    trace_term = sum(opt_trace.([model.invDₐ[index]],[Jaa,2*ιₐ*transpose(κₐΣ),-ιₐ*transpose(model.Kab[index]),- model.κₐ[index]*transpose(Jab)]))
    term_1 = -2.0*dot(model.prevη₁[index],ιₐ*model.μ[index])
    term_2 = 2.0*dot(ιₐ*model.μ[index],model.invDₐ[index]*model.κₐ[index]*model.μ[index])
    return -0.5*(trace_term+term_1+term_2)
end

function hyperparameter_online_gradient(model::OnlineVGP{<:Likelihood{T},<:Inference{T},T},ιₐ::Matrix{T},κₐΣ::Vector{Matrix{T}},Jmm::Symmetric{T,Matrix{T}},Jab::Matrix{T},Jaa::Symmetric{T,Matrix{T}},index::Integer) where {T<:Real}
    mul!(ιₐ,(Jab-model.κₐ[1]*Jmm),model.invKmm[1])
    J_q = Jaa - (ιₐ*transpose(model.Kab[1]) + model.κₐ[1]*transpose(Jab))
    trace_term = sum(sum(opt_trace.([model.invDₐ[j]],[J_q,2*ιₐ*transpose(κₐΣ[j])])) for j in 1:model.nLatent)
    term_1 = sum(-2.0*dot(model.prevη₁[j],ιₐ*model.μ[j]) for j in 1:model.nLatent)
    term_2 = sum(2.0*dot(ιₐ*model.μ[j],model.invDₐ[j]*model.κₐ[1]*model.μ[j]) for j in 1:model.nLatent)
    return -0.5*(trace_term+term_1+term_2)
end


"""Return a function computing the gradient of the ELBO given the inducing point locations"""
function inducingpoints_gradient(model::SVGP{<:Likelihood{T},<:Inference{T},T}) where {T<:Real}
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
