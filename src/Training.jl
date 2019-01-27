
"""
Function to train the given GP model, there are options to change the number of max iterations,
give a callback function that will take the model and the actual step as arguments
and give a convergence method to stop the algorithm given specific criteria
"""
function train!(model::GP;iterations::Integer=0,callback=0,Convergence=DefaultConvergence)
    if model.verbose > 0
      println("Starting training of data of $(model.nSamples) samples with $(size(model.X,2)) features $(typeof(model)<:MultiClassGPModel ? "and $(model.K) classes" : ""), using the "*model.Name*" model")
    end

    if iterations > 0 #Reset the number of iterations to a new one
        model.nEpochs = iterations
    end
    model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter::Int64 = 1; conv = Inf;

    while true #loop until one condition is matched
        try #Allow for keyboard interruption without losing the model
            updateParameters!(model) #Update all the variational parameters
            model.Trained=true
            # println(mean(model.μ[1]))
            if model.Autotuning && (iter%model.AutotuningFrequency == 0) && iter >= 3
                updateHyperParameters!(model) #Update the hyperparameters
            end
            reset_prediction_matrices!(model) #Reset predicton matrices
            if callback != 0
                callback(model,iter) #Use a callback method if put by user
            end
            # if !isa(model,BatchGPRegression)
            #     conv = Convergence(model,iter) #Check for convergence
            # else
            #     if model.verbose > 2
            #         # warn("BatchGPRegression does not need any convergence criteria")
            #     end
            #     conv = Inf
            # end
            ### Print out informations about the convergence
            if model.verbose > 2 || (model.verbose > 1  && local_iter%10==0)
                print("Iteration : $local_iter ")
            #     print("Iteration : $iter, convergence = $conv \n")
                 print("ELBO is : $(ELBO(model))")
                 print("\n")
             end
            (iter < model.nEpochs) || break; #Verify if the number of maximum iterations has been reached
            # (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
            local_iter += 1; model.inference.nIter += 1
        catch e
            if isa(e,InterruptException)
                println("Training interrupted by user");
                break;
            else
                rethrow(e)
            end
        end
    end
    if model.verbose > 0
      println("Training ended after $local_iter iterations")
    end
    computeMatrices!(model) #Compute final version of the matrices for prediction
    post_process!(model)
    if isa(model,GibbsSamplerGPC) #Compute the mean and covariance of the samples
        model.μ = squeeze(mean(hcat(model.estimate...),2),2)
        model.Σ = cov(hcat(model.estimate...),2)
    end
    model.Trained = true
end

"Update all variational parameters of the GP Model"
function updateParameters!(model::GPModel,iter::Integer)
#Function to update variational parameters
    if model.Stochastic
        model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false) #Sample nSamplesUsed indices for the minibatches
    end
    if typeof(model) <: MultiClassGPModel
        if model.KStochastic
            model.KIndices = StatsBase.sample(1:model.K,model.nClassesUsed,replace=false)
        end
    end
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model,iter);
end


"Compute of kernel matrices for the full batch GPs"
function computeMatrices!(model::FullBatchModel{T}) where T
    if model.HyperParametersUpdated
        model.Knn = Symmetric(kernelmatrix(model.X,model.kernel) + getvariance(model.kernel)*convert(T,Jittering())*I)
        model.invK = inv(model.Knn)
        model.HyperParametersUpdated = false
    end
end

"Computate of kernel matrices for the sparse GPs"
function computeMatrices!(model::SparseModel)
    if model.HyperParametersUpdated
        model.Kmm = Symmetric(kernelmatrix(model.inducingPoints,model.kernel)+jittering*I)
        # model.Kmm = Symmetric(kernelmatrix(model.inducingPoints,model.kernel)+getvariance(model.kernel)*jittering*I)
        model.invKmm = inv(model.Kmm)
    end
    if model.HyperParametersUpdated || model.Stochastic #Also when batches change
        model.Knm = kernelmatrix(model.X[model.MBIndices,:],model.inducingPoints,model.kernel)
        model.κ = model.Knm*model.invKmm
        model.Ktilde = kerneldiagmatrix(model.X[model.MBIndices,:],model.kernel)- sum(model.κ.*model.Knm,dims=2)[:]
        @assert count(model.Ktilde.<0)==0 "Ktilde has negative values"
    end
    model.HyperParametersUpdated=false
end

"Computate of kernel matrices for the linear model"
function computeMatrices!(model::LinearModel{T}) where T
    if model.HyperParametersUpdated #TODO
        model.invΣ = Matrix{T}(I/getvalue(model.noise),model.nFeatures,model.nFeatures)
        model.HyperParametersUpdated = false
    end
end

"Compute of kernel matrices for the fullbatch multiclass GPs"
function computeMatrices!(model::MultiClassGPModel{T}) where T
    if model.HyperParametersUpdated
        if model.IndependentGPs
            model.Knn[model.KIndices] .= [Symmetric(KernelModule.kernelmatrix(model.X,model.kernel[i]) + convert(T,Jittering())*I) for i in model.KIndices]
            model.invK[model.KIndices] .= inv.(model.Knn)
        else
            model.Knn .= [Symmetric(KernelModule.kernelmatrix(model.X,model.kernel[1]) + convert(T,Jittering())*I)]
            model.invK .= inv.(model.Knn)
        end
        model.HyperParametersUpdated = false
    end
end

"Compute of kernel matrices for the sparse multiclass GPs"
function computeMatrices!(model::Union{SparseMultiClass{T},SparseSoftMaxMultiClass{T},SparseLogisticSoftMaxMultiClass{T}}) where T
    if model.HyperParametersUpdated
        if model.IndependentGPs
            model.Kmm .= broadcast((points,kernel)->Symmetric(KernelModule.kernelmatrix(points,kernel)+getvariance(kernel)*convert(T,Jittering())*I),model.inducingPoints,model.kernel)
            # model.Kmm .= broadcast((points,kernel,v)->Symmetric(MLKernels.kernelmatrix(kernel,points)+convert(T,Jittering())*I)*v,model.inducingPoints,model.altkernel,model.altvar)
        else
            model.Kmm .= [Symmetric(KernelModule.kernelmatrix(model.inducingPoints[1],model.kernel[1])+convert(T,Jittering())*I)*model.altvar[1]]
        end
        model.invKmm .= inv.(model.Kmm)
    end
    #If change of hyperparameters or if stochatic
    if model.HyperParametersUpdated || model.Stochastic
        if model.IndependentGPs
            model.Knm .= broadcast((points,kernel)->KernelModule.kernelmatrix(model.X[model.MBIndices,:],points,kernel),model.inducingPoints[model.KIndices],model.kernel[model.KIndices])
            # model.Knm .= broadcast((points,kernel,v)->MLKernels.kernelmatrix(kernel,model.X[model.MBIndices,:],points)*v,model.inducingPoints[model.KIndices],model.altkernel[model.KIndices],model.altvar[model.KIndices])
            # broadcast((points,kernel,Knm)->kernelmatrix!(Knm,model.X[model.MBIndices,:],points,kernel),model.inducingPoints[model.KIndices],model.kernel[model.KIndices],model.Knm)
            model.κ .= model.Knm.*model.invKmm[model.KIndices]
            model.Ktilde .= broadcast((knm,kappa,kernel)->kerneldiagmatrix(model.X[model.MBIndices,:],kernel).+ convert(T,Jittering()) - sum(kappa.*knm,dims=2)[:],model.Knm,model.κ,model.kernel[model.KIndices])
            # model.Ktilde .= broadcast((knm,kappa,kernel,v)->(diag(MLKernels.kernelmatrix(kernel,model.X[model.MBIndices,:])) - sum(kappa.*knm,dims=2)[:])*v,model.Knm,model.κ,model.altkernel[model.KIndices],model.altvar[model.KIndices])
        else
            kernelmatrix!(model.Knm[1],model.X[model.MBIndices,:],model.inducingPoints[1],model.kernel[1])
            model.κ .= [model.Knm[1]/model.Kmm[1]]
            model.Ktilde .= [kerneldiagmatrix(model.X[model.MBIndices,:],model.kernel[1]).+jittering - sum(model.κ[1].*model.Knm[1],dims=2)[:]]
        end
        @assert sum(count.(broadcast(x->x.<0,model.Ktilde)))==0 "Ktilde has negative values"
    end
    model.HyperParametersUpdated=false
end


function reset_prediction_matrices!(model::GPModel)
    model.TopMatrixForPrediction=0;
    model.DownMatrixForPrediction=0;
end

function getInversePrior(model::LinearModel)
    return model.invΣ
end

function getInversePrior(model::FullBatchModel)
    return model.invK
end

function getInversePrior(model::SparseModel)
    return model.invKmm
end


#### Computations of the learning rates ###

function MCInit!(model::GPModel)
    if typeof(model) <: MultiClassGPModel
        model.g = [zeros(model.m*(model.m+1)) for i in 1:model.K]
        model.h = zeros(model.K)
        #Make a MC estimation using τ samples
        # model.τ[1] = 40
        for i in 1:model.τ[1]
            if model.verbose > 2
                println("MC sampling $i/$(model.τ[1])")
            end
            model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false);
            model.KIndices = collect(1:model.K)
            computeMatrices!(model);local_update!(model);
            (grad_η₁, grad_η₂) = natural_gradient(model)
            model.g = broadcast((tau,g,grad1,eta_1,grad2,eta_2)->g + vcat(grad1-eta_1,reshape(grad2-eta_2,size(grad2,1)^2))./tau,model.τ,model.g,grad_η₁,model.η₁,grad_η₂,model.η₂)
            model.h = broadcast((tau,h,grad1,eta_1,grad2,eta_2)->h + norm(vcat(grad1-eta_1,reshape(grad2-eta_2,size(grad2,1)^2)))^2/tau,model.τ,model.h,grad_η₁,model.η₁,grad_η₂,model.η₂)
        end
        model.τ .*= model.nSamplesUsed
        model.ρ_s = broadcast((g,h)->norm(g)^2/h,model.g,model.h)
        if model.KStochastic
            reinit_variational_parameters!(model) #resize the vectors for class subsampling
        end
        if model.verbose > 1
            println("$(now()): Estimation of the natural gradient for the adaptive learning rate completed")
        end
    else
        model.g = zeros(model.m*(model.m+1));
        model.h = 0;
        #Make a MC estimation using τ samples
        for i in 1:model.τ
            model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false);
            computeMatrices!(model)
            local_update!(model)
            (grad_η₁,grad_η₂) =     natural_gradient(model)
            grads = vcat(grad_η₁,reshape(grad_η₂,size(grad_η₂,1)^2))
            model.g = model.g + grads/model.τ
            model.h = model.h + norm(grads)^2/model.τ
        end
        model.ρ_s = norm(model.g)^2/model.h
        if model.verbose > 1
            println("MCMC estimation of the gradient completed")
        end
    end
end

function computeLearningRate_Stochastic!(model::GPModel,iter::Integer,grad_1,grad_2)
    if model.Stochastic
        if model.AdaptiveLearningRate
            #Using the paper on the adaptive learning rate for the SVI (update from the natural gradients)
            model.g = (1-1/model.τ)*model.g + vcat(grad_1-model.η₁,reshape(grad_2-model.η₂,size(grad_2,1)^2))./model.τ
            model.h = (1-1/model.τ)*model.h + norm(vcat(grad_1-model.η₁,reshape(grad_2-model.η₂,size(grad_2,1)^2)))^2/model.τ
            model.ρ_s = norm(model.g)^2/model.h
            model.τ = (1.0 - model.ρ_s)*model.τ + 1.0
        else
            #Simple model of time decreasing learning rate
            model.ρ_s = (iter+model.τ_s)^(-model.κ_s)
        end
    else
      #Non-Stochastic case
      model.ρ_s = 1.0
    end
end

function computeLearningRate_Stochastic!(model::MultiClassGPModel,iter::Integer,grad_1,grad_2)
    if model.Stochastic
        if model.AdaptiveLearningRate
            #Using the paper on the adaptive learning rate for the SVI (update from the natural gradients)
            model.g[model.KIndices] .= broadcast((tau,g,grad1,eta_1,grad2,eta_2)->(1-1/tau)*g + vcat(grad1-eta_1,reshape(grad2-eta_2,size(grad2,1)^2))./tau,model.τ[model.KIndices],model.g[model.KIndices],grad_1,model.η₁[model.KIndices],grad_2,model.η₂[model.KIndices])

            model.h[model.KIndices] .= broadcast((tau,h,grad1,eta_1,grad2,eta_2)->(1-1/tau)*h + norm(vcat(grad1-eta_1,reshape(grad2-eta_2,size(grad2,1)^2)))^2/tau,model.τ[model.KIndices],model.h[model.KIndices],grad_1,model.η₁[model.KIndices],grad_2,model.η₂[model.KIndices])
            # println("G : $(norm(model.g[1])), H : $(model.h[1])")
            model.ρ_s[model.KIndices] .= broadcast((g,h)->norm(g)^2/h,model.g[model.KIndices],model.h[model.KIndices])
            model.τ[model.KIndices] .= broadcast((rho,tau)->(1.0 - rho)*tau + 1.0,model.ρ_s[model.KIndices],model.τ[model.KIndices])
        else
            #Simple model of time decreasing learning rate
            model.ρ_s[model.KIndices] = [(iter+model.τ_s)^(-model.κ_s) for i in model.KIndices]
        end
    else
      #Non-Stochastic case
      model.ρ_s[model.KIndices] .= [1.0 for i in 1:model.K]
    end
    # println("rho : $(model.ρ_s[1])")
end
