
#Train function used to update the variational parameters given the training data X and y
#Possibility to put a callback function, taking the model and the iteration number as an argument
#Also one can change the convergence function

function train!(model::GPModel;iterations::Integer=0,callback=0,Convergence=DefaultConvergence)
    if model.VerboseLevel > 0
      println("Starting training of data of $(model.nSamples) samples with $(size(model.X,2)) features $(typeof(model)<:MultiClassGPModel?"and $(model.K) classes":""), using the "*model.Name*" model")
    end

    if iterations > 0 #&& iterations < model.nEpochs
        model.nEpochs = iterations
    end
    model.evol_conv = []
    if model.Stochastic && model.AdaptiveLearningRate && !model.Trained
            #If the adaptive learning rate is selected, compute a first expectation of the gradient with MCMC (if restarting training, avoid this part)
            MCInit!(model)
    end
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
        end
        if !isa(model,GPRegression)
            conv = Convergence(model,iter) #Check for convergence
        else
            if model.VerboseLevel > 2
                # warn("GPRegression does not need any convergence criteria")
            end
            conv = Inf
        end
        ### Print out informations about the convergence
        if model.VerboseLevel > 2 || (model.VerboseLevel > 1  && iter%10==0)
            print("Iteration : $iter, convergence = $conv \n")
            println("Neg. ELBO is : $(ELBO(model))")
        end
        # (iter < model.nEpochs) || break; #Verify if any condition has been broken
        (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
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

function updateParameters!(model::GPModel,iter::Integer)
#Function to update variational parameters
    if model.Stochastic
        model.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false) #Sample nSamplesUsed indices for the minibatches
        #No replacement means one points cannot be twice in the same minibatch
    end
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

function updateParameters!(model::SparseEPGPC,iter::Integer)
    computeMatrices!(model)
    x_old = model.ν[model.MBIndices]
    vNew = model.Kmm
	x_new = model.κ'
    x_newTvNew = x_new'*vNew
	x_oldTvNew = x_old'*vNew
	x_newTvNewx_old = sum(x_newTvNew' .* x_old,2)[:]
	x_oldTvNewx_old = sum(x_oldTvNew' .* x_old,2)[:]
	x_newTvNewx_new = sum(x_newTvNew' .* x_new,2)[:]

	C1 = (model.ν[model.MBIndices].^-1 - x_oldTvNewx_old).^-1

	x_newTvOldx_new = x_newTvNewx_new + x_newTvNewx_old.^2 .* C1 # defines kappa Σ\i kappa
	x_oldTvOldx_new = x_newTvNewx_old + x_oldTvNewx_old .* x_newTvNewx_old .* C1
	x_oldTmNew = x_old' * mNew
	x_newTmNew = x_new' * mNew
	C2 = model.ν[model.MBIndices] .* x_oldTmNew - model.mu[ model.MBIndices ]
	x_newTmOld = x_newTmNew + x_oldTvOldx_new .* C2 #defines kappa μ\i

	model.b[model.MBIndices] = model.Ktilde + x_newTvOldx_new + 1
	model.a[model.MBIndices] = x_newTmOld
	model.α[model.MBIndices] = model.y[model.MBIndices] ./ sqrt.(model.b[model.MBIndices]) .*
     exp.(logpdf(Normal(),model.y[MBIndices] .* model.a[model.MBIndices] ./ sqrt(model.b[model.MBIndices])) -
		logcdf(model.y[model.MBIndices] * model.a[model.MBIndices] / sqrt(model.b[model.MBIndices])))

    #Updates for ν and mu
	eta2new = (model.α[model.MBIndices].^2 + model.α[model.MBIndices] * model.a[model.MBIndices] / model.b[model.MBIndices]) *
     (1 - (model.α[model.MBIndices].^2 + model.α[model.MBIndices] .* model.a[model.MBIndices] ./ model.b[model.MBIndices]) .* x_newTvOldx_new).^-1
	eta1new = eta2new .* model.a[model.MBIndices] + model.α[model.MBIndices] + model.α[model.MBIndices] .* x_newTvOldx_new .* eta2new

    #Stochastic updates
	eta1new = (1 - model.ρ_s) *  model.mu[model.MBIndices] + model.ρ_s * eta1new
	eta2new = (1 - model.ρ_s) *  model.ν[ model.MBIndices ] + model.ρ_s * eta2new

	# This avoids uniform approximate factors

	eta2new[ abs(eta2new) .< 1e-10] = 1e-10

	# We update the posterior approximation

	if count(model.ν[ model.MBIndices ]!=0)>0
		vOld = vNew + (x_oldTvNew'*inv(diag(model.ν[ model.MBIndices ].^-1) - x_oldTvNew * x_old)) * x_oldTvNew
	else
		vOld = vNew
    end

	x_oldTvOld = x_old' * vOld
	mOld = mNew + x_oldTvOld' * ((repmat(model.mu[model.MBIndices], 1, model.m) * x_old') * mNew - model.mu[ model.MBIndices ])

	x_newTvOld = x_new' * vOld
	vNew = vOld - x_newTvOld' * inv(diag(eta2new.^-1) + x_newTvOld * x_new) * x_newTvOld
	x_newTvNew = x_new' * vNew
	mNew = mOld - x_newTvNew' * ((repmat(eta2new, 1, model.m) * x_new') * mOld - eta1new)

	model.ν[ model.MBIndices ] = eta2new
	model.mu[ model.MBIndices ] = eta1new
end

#### Computations of the kernel matrices for the different type of models ####

function computeMatrices!(model::SparseModel)
    if model.HyperParametersUpdated
        model.Kmm = Symmetric(kernelmatrix(model.inducingPoints,model.kernel)+model.noise*eye(model.nFeatures))
        model.invKmm = inv(model.Kmm)
    end
    #If change of hyperparameters or if stochatic
    if model.HyperParametersUpdated || model.Stochastic
        Knm = kernelmatrix(model.X[model.MBIndices,:],model.inducingPoints,model.kernel)
        model.κ = Knm/model.Kmm
        #println( diagkernelmatrix(model.X[model.MBIndices,:],model.kernel))
        #println(sum(model.κ.*Knm,2)[:])
        model.Ktilde = diagkernelmatrix(model.X[model.MBIndices,:],model.kernel) - sum(model.κ.*Knm,2)[:]
        #println(model.Ktilde)
        #+ model.noise*ones(length(model.MBIndices))
        @assert count(model.Ktilde.<0)==0 "Ktilde has negative values"
    end
    model.HyperParametersUpdated=false
end

function computeMatrices!(model::FullBatchModel)
    if model.HyperParametersUpdated
        model.invK = inv(Symmetric(kernelmatrix(model.X,model.kernel) + model.noise*eye(model.nFeatures)))
        model.HyperParametersUpdated = false
    end
end

function computeMatrices!(model::LinearModel)
    if model.HyperParametersUpdated
        model.invΣ =  (1.0/model.noise)*eye(model.nFeatures)
        model.HyperParametersUpdated = false
    end
end

function computeMatrices!(model::MultiClass)
    if model.HyperParametersUpdated
        model.invK = inv(Symmetric(kernelmatrix(model.X,model.kernel) + model.noise*eye(model.nFeatures)))
        model.HyperParametersUpdated = false
    end
end

function computeMatrices!(model::SparseMultiClass)
    if model.HyperParametersUpdated
        if model.KInducingPoints
            model.Kmm = broadcast(points->Symmetric(kernelmatrix(points,model.kernel)+model.noise*eye(model.nFeatures)),model.inducingPoints)
        else
            model.Kmm = [Symmetric(kernelmatrix(model.inducingPoints[1],model.kernel)+model.noise*eye(model.nFeatures))]
        end
        model.invKmm = inv.(model.Kmm)
    end
    #If change of hyperparameters or if stochatic
    if model.HyperParametersUpdated || model.Stochastic
        if model.KInducingPoints
            Knm = broadcast(points->kernelmatrix(model.X[model.MBIndices,:],points,model.kernel),model.inducingPoints)
            model.κ = Knm./model.Kmm
            model.Ktilde = broadcast((knm,kappa)->diagkernelmatrix(model.X[model.MBIndices,:],model.kernel) - sum(kappa.*knm,2)[:],Knm,model.κ)
        else
            Knm = kernelmatrix(model.X[model.MBIndices,:],model.inducingPoints[1],model.kernel)
            model.κ = [Knm/model.Kmm[1]]
            model.Ktilde = [diagkernelmatrix(model.X[model.MBIndices,:],model.kernel) - sum(model.κ[1].*Knm,2)[:]]
        end
        @assert sum(count.(broadcast(x->x.<0,model.Ktilde)))==0 "Ktilde has negative values"
    end
    model.HyperParametersUpdated=false
end

function reset_prediction_matrices!(model::GPModel)
    model.TopMatrixForPrediction=0;
    model.DownMatrixForPrediction=0;
end
#### Get Functions ####

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
                model.α[model.MBIndices] = (1 - Z*model.μ).^2 +  squeeze(sum((Z*model.ζ).*Z,2),2)+model.Ktilde;
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

function computeLearningRate_Stochastic!(model::GPModel,iter::Integer,grad_1,grad_2)
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

function computeLearningRate_Stochastic!(model::MultiClassGPModel,iter::Integer,grad_1,grad_2)
    if model.Stochastic
        if model.AdaptiveLearningRate
            #Using the paper on the adaptive learning rate for the SVI (update from the natural gradients)
            model.g = broadcast((tau,g,grad1,eta_1,grad2,eta_2)->(1-1/tau)*g + vcat(grad1-eta_1,reshape(grad2-eta_2,size(grad2,1)^2))./tau,model.τ,model.g,grad_1,model.η_1,grad_2,model.η_2)
            model.h = broadcast((tau,h,grad1,eta_1,grad2,eta_2)->(1-1/tau)*h + norm(vcat(grad1-eta_1,reshape(grad2-eta_2,size(grad2,1)^2)))^2/tau,model.τ,model.h,grad_1,model.η_1,grad_2,model.η_2)
            # println("G : $(norm(model.g[1])), H : $(model.h[1])")
            model.ρ_s = broadcast((g,h)->norm(g)^2/h,model.g,model.h)
            model.τ = broadcast((rho,tau)->(1.0 - rho)*tau + 1.0,model.ρ_s,model.τ)
        else
            #Simple model of time decreasing learning rate
            model.ρ_s = [(iter+model.τ_s)^(-model.κ_s) for i in 1:model.K]
        end
    else
      #Non-Stochastic case
      model.ρ_s = [1.0 for i in 1:model.K]
    end
    # println("rho : $(model.ρ_s[1])")
end
