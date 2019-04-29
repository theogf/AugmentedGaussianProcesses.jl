
""" `train!(model::AbstractGP;iterations::Integer=100,callback=0,conv_function=0)`

Function to train the given GP `model`.

**Keyword Arguments**

there are options to change the number of max iterations,
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `conv_function::Function` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(model::OnlineVGP,X::AbstractArray,y::AbstractArray;iterations::Integer=100,callback=0,Convergence=0)
    if model.verbose > 0
      println("Starting training $model with $(model.nSample) samples with $(size(model.X,2)) features and $(model.nLatent) latent GP"*(model.nLatent > 1 ? "s" : ""))# using the "*model.Name*" model")
    end

    @assert iterations > 0  "Number of iterations should be positive"
    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter::Int64 = 1; conv = Inf;

    while true #loop until one condition is matched
        try #Allow for keyboard interruption without losing the model
            update_parameters!(model) #Update all the variational parameters
            model.Trained = true
            if callback != 0
                callback(model,model.inference.nIter) #Use a callback method if put by user
            end
            if model.Autotuning && (model.inference.nIter%model.atfrequency == 0) && model.inference.nIter >= 3
                update_hyperparameters!(model) #Update the hyperparameters
            end
            ### Print out informations about the convergence
            if model.verbose > 2 || (model.verbose > 1  && local_iter%10==0)
                print("Iteration : $local_iter ")
                 print("ELBO is : $(ELBO(model))")
                 print("\n")
             end
            local_iter += 1; model.inference.nIter += 1
            (local_iter <= iterations) || break; #Verify if the number of maximum iterations has been reached
            isa(model,OnlineVGP) && model.Sequential && model.dataparsed && break
            # (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
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


"""Update all variational parameters of the online sparse variational GP Model"""
function update_parameters!(model::OnlineVGP)
    if model.Sequential
        model.inference.nSamplesUsed = min(model.inference.nSamplesUsed,model.nSample-model.lastindex+1)
        model.inference.MBIndices = model.lastindex:(model.lastindex+model.inference.nSamplesUsed-1) #Sample the next nSamplesUsed points
        model.lastindex += model.inference.nSamplesUsed-1
        if model.lastindex+model.inference.nSamplesUsed >= model.nSample #WARNING This exclude the last set of points!
            model.dataparsed=true #Indicate all data has been visited
        end
    else
        model.inference.MBIndices = StatsBase.sample(1:model.nSample,model.inference.nSamplesUsed,replace=false) #Sample nSamplesUsed points randomly
    end
    updateZ!(model);
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model);
end


"""Compute kernel matrices for online variational GPs"""
function computeMatrices!(model::OnlineVGP{<:Likelihood,<:Inference,T}) where {T<:Real}
    # if model.inference.HyperParametersUpdated
        model.Kmm .= broadcast((Z,kernel)->Symmetric(KernelModule.kernelmatrix(Z,kernel)+getvariance(kernel)*convert(T,Jittering())*I),model.Z,model.kernel)
        model.invKmm .= inv.(model.Kmm)
        model.Kab .= broadcast((Z,Zₐ,kernel)->kernelmatrix(Zₐ,Z,kernel),model.Z,model.Zₐ,model.kernel)
        model.κₐ .= model.Kab.*model.invKmm
        model.K̃ₐ .= kernelmatrix.(model.Zₐ,model.kernel) + model.κₐ.*transpose.(model.Kab)
        model.Knm .= kernelmatrix.([model.X[model.inference.MBIndices,:]],model.Z,model.kernel)
        model.κ .= model.Knm.*model.invKmm
        model.K̃ .= kerneldiagmatrix.([model.X[model.inference.MBIndices,:]],model.kernel) .+ [convert(T,Jittering())*ones(T,model.inference.nSamplesUsed)] - opt_diag.(model.κ,model.Knm)
        @assert sum(count.(broadcast(x->x.<0,model.K̃)))==0 "K̃ has negative values"
    # end
    model.inference.HyperParametersUpdated=false
end
