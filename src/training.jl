
"""
Function to train the given GP model, there are options to change the number of max iterations,
give a callback function that will take the model and the actual step as arguments
and give a convergence method to stop the algorithm given specific criteria
"""
function train!(model::GP;iterations::Integer=100,callback=0,Convergence=0)
    if model.verbose > 0
      println("Starting training of data of $(model.nSample) samples with $(size(model.X,2)) features and $(model.nLatent) latent GPs")# using the "*model.Name*" model")
    end

    @assert iterations > 0  "Number of iterations should be positive"
    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter::Int64 = 1; conv = Inf;

    while true #loop until one condition is matched
        try #Allow for keyboard interruption without losing the model
            update_parameters!(model) #Update all the variational parameters
            model.Trained = true
            if callback != 0
                # computeMatrices!(model) #Reupdate matrices after optimization
                callback(model,model.inference.nIter) #Use a callback method if put by user
            end
            if model.Autotuning && (model.inference.nIter%model.atfrequency == 0) && model.inference.nIter >= 3
                update_hyperparameters!(model) #Update the hyperparameters
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
            (local_iter < iterations) || break; #Verify if the number of maximum iterations has been reached
            # (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
            local_iter += 1; model.inference.nIter += 1
        catch e
            if isa(e,InterruptException)
                println("Training interrupted by user at iteration $local_iter");
                break;
            else
                rethrow(e)
            end
        end
    end
    if model.verbose > 0
      println("Training ended after $local_iter iterations. Total number of iterations $(model.inference.nIter)")
    end
    computeMatrices!(model) #Compute final version of the matrices for prediction
    post_process!(model)
    model.Trained = true
end

"Update all variational parameters of the GP Model"
function update_parameters!(model::VGP)
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model);
end

function update_parameters!(model::SVGP)
    if model.inference.Stochastic
        model.inference.MBIndices = StatsBase.sample(1:model.inference.nSamples,model.inference.nSamplesUsed,replace=false) #Sample nSamplesUsed indices for the minibatches
    end
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model);
end

"Compute of kernel matrices for variational GPs"
function computeMatrices!(model::VGP{<:Likelihood,<:Inference,T}) where {T<:Real}
    if model.inference.HyperParametersUpdated
        model.Knn .= Symmetric.(KernelModule.kernelmatrix.([model.X],model.kernel) .+ [Diagonal(convert(T,Jittering())*I,model.nFeature)])
        model.invKnn .= inv.(model.Knn)
    end
end

"Computate of kernel matrices sparse variational GPs"
function computeMatrices!(model::SVGP{<:Likelihood,<:Inference,T}) where {T<:Real}
    if model.inference.HyperParametersUpdated
        model.Kmm .= broadcast((Z,kernel)->Symmetric(KernelModule.kernelmatrix(Z,kernel)+getvariance(kernel)*convert(T,Jittering())*I),model.Z,model.kernel)
        model.invKmm .= inv.(model.Kmm)
    end
    #If change of hyperparameters or if stochatic
    if model.inference.HyperParametersUpdated || model.inference.Stochastic
        model.Knm .= broadcast((Z,kernel)->KernelModule.kernelmatrix(model.X[model.inference.MBIndices,:],Z,kernel),model.Z,model.kernel)
        model.κ .= model.Knm.*model.invKmm
        model.K̃ .= kerneldiagmatrix.([model.X[model.inference.MBIndices,:]],model.kernel) .+ [convert(T,Jittering())*ones(T,model.inference.nSamplesUsed)] - opt_diag.(model.κ,model.Knm)
        @assert sum(count.(broadcast(x->x.<0,model.K̃)))==0 "K̃ has negative values"
    end
    model.inference.HyperParametersUpdated=false
end

function MCInit!(model::GP)
    if typeof(model) <: MultiClassGPModel
        model.g = [zeros(model.m*(model.m+1)) for i in 1:model.K]
        model.h = zeros(model.K)
        #Make a MC estimation using τ samples
        # model.τ[1] = 40
        for i in 1:model.τ[1]
            if model.verbose > 2
                println("MC sampling $i/$(model.τ[1])")
            end
            model.inference.MBIndices = StatsBase.sample(1:model.nSamples,model.nSamplesUsed,replace=false);
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
