# Functions related to the Extreme Gaussian Process Classifier (XGPC)
# (paper currently being reviewed for ICML 2018)

function variablesUpdate_XGPC!(model::BatchXGPC,iter)
    model.α = sqrt.(diag(model.ζ)+model.μ.^2) #Cf derivation of updates
    θs = (1.0./(2.0*model.α)).*tanh.(model.α./2.0)
    (model.η_1,model.η_2) = naturalGradientELBO_XGPC(θs,model.y,model.invK)
    model.ζ = -0.5*inv(model.η_2); model.μ = model.ζ*model.η_1 #Back to the normal distribution parameters (needed for α updates)
end

#NEW VERSION SEEMS TO NOT WORK CORRECTLY
# function variablesUpdate_XGPC!(model::SparseXGPC,iter)
#     model.α[model.MBIndices] = sqrt.(model.Ktilde+sum((-0.5*model.κ/model.η_2).*model.κ,2)+(model.κ*model.μ).^2)
#     θs = 0.5*tanh.(0.5*model.α[model.MBIndices])./model.α[model.MBIndices]
#     (grad_η_1,grad_η_2) = naturalGradientELBO_XGPC(θs,model.y[model.MBIndices],model.invKmm; κ=model.κ,stoch_coef=model.Stochastic ? model.StochCoeff : 1.0)
#     computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
#     model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_η_1; model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_η_2 #Update of the natural parameters with noisy/full natural gradient
#     model.μ = -0.5*model.η_2\model.η_1 #Back to the distribution parameters (needed for α updates)
#     model.ζ = -0.5*Symmetric(inv(model.η_2));
# end

function variablesUpdate_XGPC!(model::SparseXGPC,iter)
    model.α[model.MBIndices] = sqrt.(model.Ktilde+sum(model.κ.*transpose(model.ζ*model.κ.'),2)+(model.κ*model.μ).^2)
    θs = 0.5*tanh.(0.5*model.α[model.MBIndices])./model.α[model.MBIndices]
    (grad_η_1,grad_η_2) = naturalGradientELBO_XGPC(θs,model.y[model.MBIndices],model.invKmm; κ=model.κ,stoch_coef=model.Stochastic ? model.StochCoeff : 1.0)
    computeLearningRate_Stochastic!(model,iter,grad_η_1,grad_η_2);
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_η_1; model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_η_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*Symmetric(inv(model.η_2)); model.μ = model.ζ*model.η_1 #Back to the distribution parameters (needed for α updates)
end

function ELBO(model::BatchXGPC)
    ELBO_v = model.nSamples*(0.5-log.(2.0)) #Constant
    θ = 1./(2*model.α).*tanh.(model.α/2.0) #Reparametrization of c
    ELBO_v += 0.5*(logdet(model.ζ)+logdet(model.invK)) #Logdet computations
    ELBO_v += -0.5*sum((model.invK+Diagonal(θ)).*transpose(model.ζ+model.μ*transpose(model.μ))) #Computation of the trace
    ELBO_v += 0.5*dot(model.y,model.μ)
    ELBO_v += sum(0.5*(model.α.^2).*θ-log.(cosh.(0.5*model.α)))
    return -ELBO_v
end

function ELBO(model::SparseXGPC)
    model.StochCoeff = model.nSamples/model.nSamplesUsed
    ELBO_v = -model.nSamples*log(2)+model.m/2.0
    θ = 1./(2*model.α[model.MBIndices]).*tanh.(model.α[model.MBIndices]/2.0)
    ELBO_v += 0.5*(logdet(model.ζ)+logdet(model.invKmm))
    ELBO_v += -0.5*(sum((model.invKmm+model.StochCoeff*model.κ'*Diagonal(θ)*model.κ).*transpose(model.ζ+model.μ*transpose(model.μ))))
    ELBO_v += model.StochCoeff*0.5*dot(model.y[model.MBIndices],model.κ*model.μ)
    ELBO_v += -0.5*model.StochCoeff*dot(θ,model.Ktilde)
    ELBO_v += model.StochCoeff*sum(0.5*(model.α[model.MBIndices].^2).*θ-log.(cosh.(0.5*model.α[model.MBIndices])))
    return -ELBO_v
end

function naturalGradientELBO_XGPC(θ,y,invPrior;κ=0,stoch_coef::Float64=1.0)
    if κ == 0
        #Full batch case
        grad_1 =  0.5*y
        grad_2 = -0.5*(Diagonal(θ) + invPrior)
    else
        #Sparse case
        grad_1 =  0.5*stoch_coef*κ'*y
        grad_2 = -0.5*(stoch_coef*transpose(κ)*Diagonal(θ)*κ + invPrior)
  end
  return (grad_1,grad_2)
end


function hyperparameter_gradient_function(model::SparseXGPC)
    #General values used for all gradients
    B = model.μ*transpose(model.μ) + model.ζ
    Kmn = kernelmatrix(model.inducingPoints,model.X[model.MBIndices,:],model.kernel)
    Θ = Diagonal(0.25./model.α[model.MBIndices].*tanh.(0.5*model.α[model.MBIndices]))
    return function(Js)
                Jmm = Js[1]; Jnm = Js[2]; Jnn = Js[3];
                ι = (Jnm-model.κ*Jmm)*model.invKmm
                Jtilde = Jnn - sum(ι.*(Kmn.'),2) - sum(model.κ.*Jnm,2)
                V = model.invKmm*Jmm
                return 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*Θ*model.κ + model.κ'*Θ*ι)) .* transpose(B)) - trace(V) - model.StochCoeff*dot(diag(Θ),Jtilde)
                    + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
     end
end
function inducingpoints_gradient(model::SparseXGPC)
    gradients_inducing_points = zeros(model.inducingPoints)
    B = model.μ*transpose(model.μ) + model.ζ
    Kmn = kernelmatrix(model.inducingPoints,model.X[model.MBIndices,:],model.kernel)
    Θ = Diagonal(0.25./model.α[model.MBIndices].*tanh.(0.5*model.α[model.MBIndices]))
    for i in 1:model.m #Iterate over the points
        Jnm,Jmm = computeIndPointsJ(model,i)
        for j in 1:model.nDim #Compute the gradient over the dimensions
            ι = (Jnm[j,:,:]-model.κ*Jmm[j,:,:])*model.invKmm
            Jtilde = -sum(ι.*(Kmn.'),2)[:]-sum(model.κ.*Jnm[j,:,:],2)[:]
            V = model.invKmm*Jmm[j,:,:]
            gradients_inducing_points[i,j] = 0.5*(sum( (V*model.invKmm - model.StochCoeff*(ι'*Θ*model.κ + model.κ'*Θ*ι)) .* transpose(B)) - trace(V) - model.StochCoeff*dot(diag(Θ),Jtilde)
                + model.StochCoeff*dot(model.y[model.MBIndices],ι*model.μ))
        end
    end
    return gradients_inducing_points
end
