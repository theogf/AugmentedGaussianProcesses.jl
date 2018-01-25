 # include("../src/DataAugmentedModels.jl")

function train!(model::AugmentedModel;iterations::Integer=0,callback=0,Convergence=DefaultConvergence)
    if model.VerboseLevel > 0
      println("Starting training of data of size $((model.nSamples,size(model.X,2))), using the "*model.Name*" model")
    end

    if iterations > 0 #&& iterations < model.nEpochs
        model.nEpochs = iterations
    end
    model.evol_conv = []
    if model.Stochastic
        if model.AdaptiveLearningRate #If the adaptive learning rate is selected, compute a first expectation of the gradient with MCMC
            model.g = zeros(model.m*(model.m+1));
            model.h = 0;
            for i in 1:model.τ
                model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false);
                computeMatrices!(model)
                if model.ModelType==BSVM
                    Z = Diagonal(model.y[model.MBIndices])*model.κ;
                    model.α[model.MBIndices] = (1 - Z*model.μ).^2 +  squeeze(sum((Z*model.ζ).*Z,2),2)+model.Ktilde;
                    (grad_η_1,grad_η_2) = naturalGradientELBO_BSVM(model.α[model.MBIndices],Z, model.invKmm, model.StochCoeff)
                elseif model.ModelType==XGPC
                    model.α[model.MBIndices] = sqrt.(model.Ktilde+diag(model.κ*model.ζ*model.κ')+(model.κ*model.μ).^2)
                    θs = (1.0./(2.0*model.α[model.MBIndices])).*tanh.(model.α[model.MBIndices]./2.0)
                    (grad_η_1,grad_η_2) = naturalGradientELBO_XGPC(θs,model.y[model.MBIndices],model.invKmm; κ=model.κ,stoch_coef=model.StochCoeff)
                end
                model.g = model.g + 1/model.τ*vcat(grad_η_1,reshape(grad_η_2,size(grad_η_2,1)^2))
                model.h = model.h + 1/model.τ*norm(vcat(grad_η_1,reshape(grad_η_2,size(grad_η_2,1)^2)))^2
            end


        end
    end
    computeMatrices!(model)
    model.Trained = true
    iter::Int64 = 1; conv = Inf;
    while true #do while loop

        updateParameters!(model,iter) #Update all the variational parameters
        if callback != 0
                callback(model,iter) #Use a callback method if put by user
        end
        # model.Trained = true #Confirm model has been initialized (and can make prediction)

        if model.Autotuning && (iter%model.AutotuningFrequency == 0) && iter >= 3
            updateHyperParameters!(model,iter) #Do the hyper-parameter optimization
            computeMatrices!(model)
            # callback(model,iter;hyper=true)
        end
        conv = Convergence(model,iter) #Check for convergence
        ### Print out informations about the convergence
        if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%10==0)
            print("Iteration : $iter, convergence = $conv \n")
        end
        (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
        iter += 1;
    end
    if model.VerboseLevel > 0
      println("Training ended after $iter iterations")
    end
    #Compute final version of the matrices for prediction
    if isa(model,GibbsSamplerGPC) #Compute the average of the samples
        model.μ = squeeze(mean(hcat(model.estimate...),2),2)
        model.ζ = cov(hcat(model.estimate...),2)
    end
    computeMatrices!(model)
    model.Trained = true
end

function updateParameters!(model::AugmentedModel,iter::Integer)
#Function to update variational parameters
    if model.Stochastic
        model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false) #Sample nSamplesUsed indices for the minibatches
        #No replacement means one points cannot be twice in the same minibatch
    end
    computeMatrices!(model); #Compute the matrices if necessary
    if model.ModelType == BSVM
        variablesUpdate_BSVM!(model,iter)
    elseif model.ModelType == XGPC
        variablesUpdate_XGPC!(model,iter)
    end
end

function updateParameters!(model::GibbsSamplerGPC,iter::Integer)
#Sample for every parameter
    computeMatrices!(model)
    model.α = broadcast(model.pgsampler.draw,1.0,model.μ)
    push!(model.samplehistory,:ω,iter,model.α)
    C = Matrix(Symmetric(inv(diagm(model.α)+model.invK),:U))
    model.μ = rand(MvNormal(0.5*C*model.y,C))
    push!(model.samplehistory,:f,iter,model.μ)
    if iter > model.burninsamples && (iter-model.burninsamples)%model.samplefrequency==0
        push!(model.estimate,model.μ)
    end
end

#### Computations of the learning rates ###

function computeLearningRate_Stochastic!(model::AugmentedModel,iter::Integer,grad_1,grad_2)
    if model.Stochastic
        if model.AdaptiveLearningRate
            #Using the paper on the adaptive learning rate for the SVI (update from the natural gradients)
            model.g = (1-1/model.τ)*model.g + vcat(grad_1-model.η_1,reshape(grad_2-model.η_2,size(grad_2,1)^2))./model.τ
            model.h = (1-1/model.τ)*model.h + norm(vcat(grad_1-model.η_1,reshape(grad_2-model.η_2,size(grad_2,1)^2)))^2/model.τ
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

function computeLearningRate_Hyperparameter!(model::AugmentedModel,iter::Integer,gradients)
    nHyperParameters = length(model.ρ_AT);
    for i in 1:nHyperParameters
        #model.ρ_AT[i] = (iter+model.τ_AT)^(-model.κ_AT);
        model.ρ_AT[i] = model.ρ_AT[i]; #no dependency in time
    end
end
