# Functions related to the Efficient Gaussian Process Classifier (XGPC)
# https://arxiv.org/abs/1802.06383
"Update the local variational parameters of the full batch GP XGPC"
function local_update!(model::BatchXGPC)
    model.α = sqrt.(diag(model.Σ)+model.μ.^2)
end

"Compute the variational updates for the full GP XGPC"
function variational_updates!(model::BatchXGPC,iter)
    local_update!(model)
    θ = 0.5*tanh.(0.5*model.α)./model.α
    natural_gradient_XGPC(model)
    global_update!(model)
end

"Update the local variational parameters of the sparse GP XGPC"
function local_update!(model::SparseXGPC)
    model.α = sqrt.(model.Ktilde+sum((model.κ*model.Σ).*model.κ,dims=2)[:]+(model.κ*model.μ).^2)
    model.θ = 0.5*tanh.(0.5*model.α)./model.α

end

"Compute the variational updates for the sparse GP XGPC"
function variational_updates!(model::SparseXGPC,iter::Integer)
    local_update!(model)
    (grad_η_1,grad_η_2) = natural_gradient_XGPC(model)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

"Update the local variational parameters of the online GP XGPC"
function local_update!(model::OnlineXGPC)
    model.α = sqrt.(model.Ktilde+sum((model.κ*model.Σ).*model.κ,dims=2)[:]+(model.κ*model.μ).^2)
end

"Compute the variational updates for the online GP XGPC"
function variational_updates!(model::OnlineXGPC,iter::Integer)
    local_update!(model)
    θ = 0.5*tanh.(0.5*model.α)./model.α
    (grad_η_1,grad_η_2) = natural_gradient_XGPC(model)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    global_update!(model,grad_η_1,grad_η_2)
end

"Return the natural gradients of the ELBO given the natural parameters"
function natural_gradient_XGPC(model::BatchXGPC)
    model.η_1 =  0.5*model.y
    model.η_2 = -0.5*(Diagonal{Float64}(model.θ) + model.invK)
end

"Return the natural gradients of the ELBO given the natural parameters"
function natural_gradient_XGPC(model::SparseXGPC)
    grad_1 =  0.5*model.StochCoeff*model.κ'*model.y[model.MBIndices]
    grad_2 = -0.5.*(model.StochCoeff*transpose(model.κ)*Diagonal{Float64}(model.θ)*model.κ .+ model.invKmm)
    return (grad_1,grad_2)
end


"Compute the negative ELBO for the full batch XGPC Model"
function ELBO(model::BatchXGPC)
    ELBO_v = model.nSamples*(0.5-log.(2.0)) #Constant
    θ = 1.0./(2*model.α).*tanh.(model.α/2.0) #Computation of mean of ω
    ELBO_v += 0.5*(logdet(model.Σ)+logdet(model.invK)) #Logdet computations
    ELBO_v += -0.5*sum((model.invK+Diagonal(θ)).*transpose(model.Σ+model.μ*transpose(model.μ))) #Computation of the trace
    ELBO_v += 0.5*dot(model.y,model.μ)
    ELBO_v += sum(0.5*(model.α.^2).*θ-log.(cosh.(0.5*model.α)))
    return -ELBO_v
end

"Compute the negative ELBO for the sparse XGPC Model"
function ELBO(model::SparseXGPC)
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    ELBO_v = -model.nSamples*log(2)+model.m/2.0
    θ = 1.0./(2*model.α).*tanh.(model.α/2.0)
    ELBO_v += 0.5*(logdet(model.Σ)+logdet(model.invKmm))
    ELBO_v += -0.5*(sum((model.invKmm+model.StochCoeff*model.κ'*Diagonal(θ)*model.κ).*transpose(model.Σ+model.μ*transpose(model.μ))))
    ELBO_v += model.StochCoeff*0.5*dot(model.y[model.MBIndices],model.κ*model.μ)
    ELBO_v += -0.5*model.StochCoeff*dot(θ,model.Ktilde)
    ELBO_v += model.StochCoeff*sum(0.5*(model.α.^2).*θ-log.(cosh.(0.5*model.α)))
    return -ELBO_v
end

"Compute the negative ELBO for the sparse XGPC Model"
function ELBO(model::OnlineXGPC)
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    ELBO_v = -model.nSamples*log(2)+model.m/2.0
    θ = 1.0./(2*model.α).*tanh.(model.α/2.0)
    ELBO_v += 0.5*(logdet(model.Σ)+logdet(model.invKmm))
    ELBO_v += -0.5*(sum((model.invKmm+model.StochCoeff*model.κ'*Diagonal(θ)*model.κ).*transpose(model.Σ+model.μ*transpose(model.μ))))
    ELBO_v += model.StochCoeff*0.5*dot(model.y[model.MBIndices],model.κ*model.μ)
    ELBO_v += -0.5*model.StochCoeff*dot(θ,model.Ktilde)
    ELBO_v += model.StochCoeff*sum(0.5*(model.α.^2).*θ-log.(cosh.(0.5*model.α)))
    return -ELBO_v
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters for a XGPC Model"
function hyperparameter_gradient_function(model::SparseXGPC)
    B = model.μ*transpose(model.μ) + model.Σ
    Kmn = kernelmatrix(model.inducingPoints,model.X[model.MBIndices,:],model.kernel)
    Θ = Diagonal(0.25./model.α.*tanh.(0.5*model.α))
    return function(Js,i,j)
                Jmm = Js[1]; Jnm = Js[2]; Jnn = Js[3];
                ι = (Jnm-model.κ*Jmm)*model.invKmm
                Jtilde = Jnn - sum(ι.*(transpose(Kmn)),dims=2) - sum(model.κ.*Jnm,dims=2)
                V = model.invKmm*Jmm
                return 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*Θ*model.κ + model.κ'*Θ*ι)) .* transpose(B)) - tr(V) - model.StochCoeff*dot(diag(Θ),Jtilde)
                    + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
     end
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters"
function inducingpoints_gradient(model::SparseXGPC)
    gradients_inducing_points = zeros(model.inducingPoints)
    B = model.μ*transpose(model.μ) + model.Σ
    Kmn = kernelmatrix(model.inducingPoints,model.X[model.MBIndices,:],model.kernel)
    Θ = Diagonal(0.25./model.α.*tanh.(0.5*model.α))
    for i in 1:model.m #Iterate over the points
        Jnm,Jmm = computeIndPointsJ(model,i)
        for j in 1:model.nDim #Compute the gradient over the dimensions
            ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])/model.Kmm
            Jtilde = -sum(ι.*(transpose(Kmn)),dims=2)[:]-sum(model.κ.*Jnm[j,:,:],dims=2)[:]
            V = model.Kmm\Jmm[j,:,:]
            gradients_inducing_points[i,j] = 0.5*(sum( (V/model.Kmm - model.StochCoeff*(ι'*Θ*model.κ + model.κ'*Θ*ι)) .* transpose(B)) - tr(V) - model.StochCoeff*dot(diag(Θ),Jtilde)
                + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
        end
    end
    return gradients_inducing_points
end
