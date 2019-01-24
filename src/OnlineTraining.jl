
"""
Function to train the given Online GP model, there are options to change the number of max iterations,
give a callback function that will take the model and the actual step as arguments
and give a convergence method to stop the algorithm given specific criteria
"""
function train!(model::OnlineGPModel;iterations::Integer=0,callback=0,convergence=DefaultConvergence)
    if model.verbose > 0
      println("Starting training of online training of data with $(model.nSamples) samples with $(size(model.X,2)) features :, using the "*model.Name*" model")
    end

    if iterations > 0 #&& iterations < model.nEpochs
        model.nEpochs = iterations
    end
    # model.evol_conv = []
    ##TODO for now it is not possible to compute an adaptive learning rate
    # if model.Stochastic && model.AdaptiveLearningRate && !model.Trained
            #If the adaptive learning rate is selected, compute a first expectation of the gradient with MCMC (if restarting training, avoid this part)
            # MCInit!(model)
    # end
    computeMatrices!(model)
    model.Trained = true
    iter::Int64 = 1; conv = Inf;
    while true #do while loop
        if callback != 0
                callback(model,iter) #Use a callback method if put by user
        end
        updateParameters!(model,iter) #Update all the variational parameters
        reset_prediction_matrices!(model) #Reset predicton matrices
        if model.Autotuning && (iter%model.AutotuningFrequency == 0) && iter >= 3
            updateHyperParameters!(model) #Do the hyper-parameter optimization
            computeMatrices!(model)
        end
        # conv = convergence(model,iter) #Check for convergence
        ### Print out informations about the convergence
         if model.verbose > 2 || (model.verbose > 1  && iter%10==0)
            print("Iteration : $iter, convergence = $conv \n")
            # println("Neg. ELBO is : $(ELBO(model))")
        end
        # (iter < model.nEpochs) || break; #Verify if any condition has been broken
         ((iter < model.nEpochs && !model.alldataparsed) && conv > model.ϵ) || break; #Verify if any condition has been broken
        iter += 1;
    end
    if model.verbose > 0
      println("Training ended after $iter iterations")
    end
    computeMatrices!(model) #Recompute matrices if hyperparameters have been changed
    #Compute final version of the matrices for prediction
    if isa(model,GibbsSamplerGPC) #Compute the average of the samples
        model.μ = squeeze(mean(hcat(model.estimate...),2),2)
        model.Σ = cov(hcat(model.estimate...),2)
    elseif isa(model,MultiClass) || isa(model,SparseMultiClass)
        model.Σ = broadcast(x->(-0.5*inv(x)),model.η₂)
    elseif !isa(model,BatchGPRegression)
        model.Σ = -0.5*inv(model.η₂);
    end
    computeMatrices!(model)
    model.Trained = true
end

"""
    Subfunction to update all the parameters (except for hyperparameters of the model)
"""
function updateParameters!(model::OnlineGPModel,iter::Integer)
    #Select a new batch of data given the method choice
    if model.Sequential
        newbatchsize = min(model.nSamplesUsed-1,model.nSamples-model.lastindex)
        model.MBIndices = model.lastindex:(model.lastindex+newbatchsize) #Sample the next nSamplesUsed points
        model.lastindex += newbatchsize
        if newbatchsize < (model.nSamplesUsed-1)
            model.alldataparsed=true #Indicate all data has been visited
        end
    else
        model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false) #Sample nSamplesUsed points randomly
    end
    update_points!(model) #Update the location of the inducing points
    computeMatrices!(model); #Recompute the matrices given the new batch of data and inducing points
    #Update the variational parameters given the type of model
    if model.ModelType == BSVM
        variablesUpdate_BSVM!(model,iter)
    elseif model.ModelType == XGPC
        variablesUpdate_XGPC!(model,iter)
    elseif model.ModelType == Regression
        variablesUpdate_Regression!(model,iter)
    elseif typeof(model) <: MultiClassGPModel
        variablesUpdate_MultiClass!(model,iter)
    end
end

"""
Subfunction to update the inducing points in an online manner
"""
function update_points!(model::OnlineGPModel)
    update!(model.kmeansalg,model.X[model.MBIndices,:],model.y[model.MBIndices],model)
    NCenters = model.kmeansalg.k
    Nnewpoints = NCenters-model.m
    #Make the latent variables larger #TODO Preallocating them might be a better option
    if Nnewpoints!=0
        model.μ = vcat(model.μ, zeros(Nnewpoints))
        model.η₁ = vcat(model.η₁, zeros(Nnewpoints))
        Σ_temp = Matrix{Float64}(I,NCenters,NCenters)
        Σ_temp[1:model.m,1:model.m] = model.Σ
        model.Σ = Σ_temp
        η₂temp = Matrix{Float64}(-0.5*I,NCenters,NCenters)
        η₂temp[1:model.m,1:model.m] = model.η₂
        model.η₂ = η₂temp
        model.m = NCenters
        model.nFeatures = model.m
    end
    model.indpoints_updated = true
end

"""
Computate all necessary kernel matrices
"""
function computeMatrices!(model::OnlineGPModel)
    if model.HyperParametersUpdated || model.indpoints_updated
        model.Kmm = Symmetric(kernelmatrix(model.kmeansalg.centers,model.kernel)+getvariance(model.kernel)*convert(T,Jittering())*I)
        model.invKmm = inv(model.Kmm)
        Knm = kernelmatrix(model.X[model.MBIndices,:],model.kmeansalg.centers,model.kernel)
        model.κ = Knm/model.Kmm
        model.Ktilde = diagkernelmatrix(model.X[model.MBIndices,:],model.kernel) - sum(model.κ.*Knm,dims=2)[:]
        @assert count(model.Ktilde.<0)==0 "Ktilde has negative values"
    end
    model.HyperParametersUpdated = false;
    model.indpoints_updated = false;
end

"""
Reset prediction matrices when any model parameter has been changed
"""
function reset_prediction_matrices!(model::OnlineGPModel)
    model.TopMatrixForPrediction=0;
    model.DownMatrixForPrediction=0;
end


"""
TODO, Compute a MCMC estimation of the natural gradient at the initialization of the model for an optimal learning rate
"""
function MCInit!(model::OnlineGPModel)
    if typeof(model) <: MultiClassGPModel
        model.g = [zeros(model.m*(model.m+1)) for i in 1:model.K]
        model.h = zeros(model.K)
        #Make a MC estimation using τ samples
        for i in 1:model.τ[1]
            model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false);
            computeMatrices!(model)
            local_updates!(model)
            (grad_η₁, grad_η₂) = naturalGradientELBO_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invKmm,model.γ,stoch_coeff=model.StochCoeff,MBIndices=model.MBIndices,κ=model.κ)
            model.g = broadcast((tau,g,grad1,eta_1,grad2,eta_2)->g + vcat(grad1-eta_1,reshape(grad2-eta_2,size(grad2,1)^2))./tau,model.τ,model.g,grad_η₁,model.η₁,grad_η₂,model.η₂)
            model.h = broadcast((tau,h,grad1,eta_1,grad2,eta_2)->h + norm(vcat(grad1-eta_1,reshape(grad2-eta_2,size(grad2,1)^2)))^2/tau,model.τ,model.h,grad_η₁,model.η₁,grad_η₂,model.η₂)
        end
        model.ρ_s = broadcast((g,h)->norm(g)^2/h,model.g,model.h)
        if model.verbose > 2
            println("$(now()): MCMC estimation of the gradient completed")
        end
    else
        model.g = zeros(model.m*(model.m+1));
        model.h = 0;
        #Make a MC estimation using τ samples
        for i in 1:model.τ #TODO Correct all this!
            model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false);
            computeMatrices!(model)
            # local_updates!(model)
            if model.ModelType==BSVM
                Z = Diagonal(model.y[model.MBIndices])*model.κ;
                model.α[model.MBIndices] = (1 - Z*model.μ).^2 +  squeeze(sum((Z*model.Σ).*Z,dims=2),2)+model.Ktilde;
                (grad_η₁,grad_η₂) = naturalGradientELBO_BSVM(model.α[model.MBIndices],Z, model.invKmm, model.StochCoeff)
            elseif model.ModelType==XGPC
                model.α[model.MBIndices] = sqrt.(model.Ktilde+diag(model.κ*model.Σ*model.κ')+(model.κ*model.μ).^2)
                θs = (1.0./(2.0*model.α[model.MBIndices])).*tanh.(model.α[model.MBIndices]./2.0)
                (grad_η₁,grad_η₂) = naturalGradientELBO_XGPC(θs,model.y[model.MBIndices],model.invKmm; κ=model.κ,stoch_coef=model.StochCoeff)
            elseif model.ModelType==Regression
                (grad_η₁,grad_η₂) = naturalGradientELBO_Regression(model.y[model.MBIndices],model.κ,model.noise,stoch_coeff=model.StochCoeff)
            end
            model.g = model.g + 1/model.τ*vcat(grad_η₁,reshape(grad_η₂,size(grad_η₂,1)^2))
            model.h = model.h + 1/model.τ*norm(vcat(grad_η₁,reshape(grad_η₂,size(grad_η₂,1)^2)))^2
        end
        model.ρ_s = norm(model.g)^2/model.h
        if model.verbose > 2
            println("MCMC estimation of the gradient completed")
        end
    end
end
