"Update the global variational parameters of the linear models"
function global_update!(model::LinearModel,grad_1::Vector,grad_2::Matrix)
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_1;
    model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_2 #Update of the natural parameters with noisy/full natural gradient
    model.μ = -0.5*model.η_2\model.η_1 #Back to the normal distribution parameters (needed for α updates)
    model.ζ = -0.5*inv(model.η_2);
end

"Update the global variational parameters of the linear models"
function global_update!(model::FullBatchModel)
    model.ζ = -0.5*inv(model.η_2);
    model.μ = model.ζ*model.η_1 #Back to the normal distribution parameters (needed for α updates)
end

"Update the global variational parameters of the sparse GP models"
function global_update!(model::SparseModel,grad_1::Vector,grad_2::Matrix)
    model.η_1 = (1.0-model.ρ_s)*model.η_1 + model.ρ_s*grad_1;
    model.η_2 = (1.0-model.ρ_s)*model.η_2 + model.ρ_s*grad_2 #Update of the natural parameters with noisy/full natural gradient
    model.ζ = -0.5*inv(model.η_2);
    model.μ = model.ζ*model.η_1 #Back to the normal distribution parameters (needed for α updates)
end

"Return a function computing the gradient of the ELBO given the kernel hyperparameters for full batch Models"
function hyperparameter_gradient_function(model::FullBatchModel)
    A = model.invK*(model.ζ+model.µ*transpose(model.μ))-Diagonal{Float64}(I,model.nSamples)
    return function(Js)
                V = model.invK*Js[1]
                return 0.5*sum(V.*transpose(A))
            end
end
