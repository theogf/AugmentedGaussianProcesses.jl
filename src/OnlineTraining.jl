
#Train function used to update the variational parameters given the training data X and y
#Possibility to put a callback function, taking the model and the iteration number as an argument
#Also one can change the convergence function

function train!(model::OnlineGPModel;iterations::Integer=0,callback=0,Convergence=DefaultConvergence)
    if model.VerboseLevel > 0
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
        # conv = Convergence(model,iter) #Check for convergence
        ### Print out informations about the convergence
         if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%10==0)
            print("Iteration : $iter, convergence = $conv \n")
            # println("Neg. ELBO is : $(ELBO(model))")
        end
        # (iter < model.nEpochs) || break; #Verify if any condition has been broken
         ((iter < model.nEpochs && !model.alldataparsed) && conv > model.ϵ) || break; #Verify if any condition has been broken
        iter += 1;
    end
    if model.VerboseLevel > 0
      println("Training ended after $iter iterations")
    end
    computeMatrices!(model)
    #Compute final version of the matrices for prediction
    if isa(model,GibbsSamplerGPC) #Compute the average of the samples
        model.μ = squeeze(mean(hcat(model.estimate...),2),2)
        model.ζ = cov(hcat(model.estimate...),2)
    elseif isa(model,MultiClass) || isa(model,SparseMultiClass)
        model.ζ = broadcast(x->(-0.5*inv(x)),model.η_2)
    elseif !isa(model,GPRegression)
        model.ζ = -0.5*inv(model.η_2);
    end
    computeMatrices!(model)
    model.Trained = true
end

function updateParameters!(model::OnlineGPModel,iter::Integer)
#Function to update variational parameters
    if model.Sequential
        newbatchsize = min(model.nSamplesUsed-1,model.nSamples-model.lastindex)
        model.MBIndices = model.lastindex:(model.lastindex+newbatchsize) #Sample nSamplesUsed indices for the minibatches
        model.lastindex += newbatchsize
        if newbatchsize < (model.nSamplesUsed-1)
            model.alldataparsed=true
        end
    else
        model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false) #Sample nSamplesUsed indices for the minibatches
    end
    update_points!(model)
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
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


function update_points!(model::OnlineGPModel)
    update!(model.kmeansalg,model.X[model.MBIndices,:],model.y[model.MBIndices],model)
    NCenters = model.kmeansalg.k
    Nnewpoints = NCenters-model.m
    #Make the latent variables larger #TODO Preallocating them might be a better option
    if Nnewpoints!=0
        model.μ = vcat(model.μ, zeros(Nnewpoints))
        model.η_1 = vcat(model.η_1, zeros(Nnewpoints))
        ζ_temp = Matrix{Float64}(I,NCenters,NCenters)
        ζ_temp[1:model.m,1:model.m] = model.ζ
        model.ζ = ζ_temp
        η_2temp = Matrix{Float64}(-0.5*I,NCenters,NCenters)
        η_2temp[1:model.m,1:model.m] = model.η_2
        model.η_2 = η_2temp
        model.m = NCenters
        model.nFeatures = model.m
    end
    model.indpoints_updated = true
end

#### Computations of the kernel matrices for the different type of models ####
function computeMatrices!(model::OnlineGPModel)
    if model.HyperParametersUpdated || model.indpoints_updated
        model.Kmm = Symmetric(kernelmatrix(model.kmeansalg.centers,model.kernel)+Diagonal{Float64}(model.noise*I,model.m))
        model.invKmm = inv(model.Kmm)
        Knm = kernelmatrix(model.X[model.MBIndices,:],model.kmeansalg.centers,model.kernel)
        model.κ = Knm/model.Kmm
        model.Ktilde = diagkernelmatrix(model.X[model.MBIndices,:],model.kernel) - sum(model.κ.*Knm,dims=2)[:]
        @assert count(model.Ktilde.<0)==0 "Ktilde has negative values"
    end
    model.HyperParametersUpdated = false;
    model.indpoints_updated = false;
end

function reset_prediction_matrices!(model::OnlineGPModel)
    model.TopMatrixForPrediction=0;
    model.DownMatrixForPrediction=0;
end


#### Computations of the learning rates ###

function MCInit!(model::OnlineGPModel)
    if typeof(model) <: MultiClassGPModel
        model.g = [zeros(model.m*(model.m+1)) for i in 1:model.K]
        model.h = zeros(model.K)
        #Make a MC estimation using τ samples
        for i in 1:model.τ[1]
            model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false);
            computeMatrices!(model)
            local_updates!(model)
            (grad_η_1, grad_η_2) = naturalGradientELBO_MultiClass(model.Y,model.θ[1],model.θ[2:end],model.invKmm,model.γ,stoch_coeff=model.StochCoeff,MBIndices=model.MBIndices,κ=model.κ)
            model.g = broadcast((tau,g,grad1,eta_1,grad2,eta_2)->g + vcat(grad1-eta_1,reshape(grad2-eta_2,size(grad2,1)^2))./tau,model.τ,model.g,grad_η_1,model.η_1,grad_η_2,model.η_2)
            model.h = broadcast((tau,h,grad1,eta_1,grad2,eta_2)->h + norm(vcat(grad1-eta_1,reshape(grad2-eta_2,size(grad2,1)^2)))^2/tau,model.τ,model.h,grad_η_1,model.η_1,grad_η_2,model.η_2)
        end
        model.ρ_s = broadcast((g,h)->norm(g)^2/h,model.g,model.h)
        if model.VerboseLevel > 2
            println("$(now()): MCMC estimation of the gradient completed")
        end
    else
        model.g = zeros(model.m*(model.m+1));
        model.h = 0;
        #Make a MC estimation using τ samples
        for i in 1:model.τ
            model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false);
            computeMatrices!(model)
            # local_updates!(model)
            if model.ModelType==BSVM
                Z = Diagonal(model.y[model.MBIndices])*model.κ;
                model.α[model.MBIndices] = (1 - Z*model.μ).^2 +  squeeze(sum((Z*model.ζ).*Z,dims=2),2)+model.Ktilde;
                (grad_η_1,grad_η_2) = naturalGradientELBO_BSVM(model.α[model.MBIndices],Z, model.invKmm, model.StochCoeff)
            elseif model.ModelType==XGPC
                model.α[model.MBIndices] = sqrt.(model.Ktilde+diag(model.κ*model.ζ*model.κ')+(model.κ*model.μ).^2)
                θs = (1.0./(2.0*model.α[model.MBIndices])).*tanh.(model.α[model.MBIndices]./2.0)
                (grad_η_1,grad_η_2) = naturalGradientELBO_XGPC(θs,model.y[model.MBIndices],model.invKmm; κ=model.κ,stoch_coef=model.StochCoeff)
            elseif model.ModelType==Regression
                (grad_η_1,grad_η_2) = naturalGradientELBO_Regression(model.y[model.MBIndices],model.κ,model.noise,stoch_coeff=model.StochCoeff)
            end
            model.g = model.g + 1/model.τ*vcat(grad_η_1,reshape(grad_η_2,size(grad_η_2,1)^2))
            model.h = model.h + 1/model.τ*norm(vcat(grad_η_1,reshape(grad_η_2,size(grad_η_2,1)^2)))^2
        end
        model.ρ_s = norm(model.g)^2/model.h
        if model.VerboseLevel > 2
            println("MCMC estimation of the gradient completed")
        end
    end
end
