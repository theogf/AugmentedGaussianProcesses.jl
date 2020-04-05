
""" `train!(model::AbstractGP;iterations::Integer=100,callback=0,conv_function=0)`

Function to train the given GP `model`.

**Keyword Arguments**

there are options to change the number of max iterations,
- `iterations::Int` : Number of iterations (not necessarily epochs!)for training
- `callback::Function` : Callback function called at every iteration. Should be of type `function(model,iter) ...  end`
- `conv_function::Function` : Convergence function to be called every iteration, should return a scalar and take the same arguments as `callback`
"""
function train!(
    m::OnlineSVGP,
    X::AbstractArray,
    y::AbstractArray;
    iterations::Int = 2,
    callback::Union{Nothing,Function} = nothing,
    Convergence = 0,
)

    m.X = wrap_X(X)
    m.y, _nLatent, m.likelihood = check_data!(m.X, y, m.likelihood)

    @assert _nLatent == m.nLatent "Data should always contains the same number of outputs"
    @assert iterations > 0 "Number of iterations should be positive"
    setnMinibatch!(m.inference, size(X, 1))
    setnSamples!(m.inference, size(X, 1))
    m.inference.MBIndices = [collect(1:size(X, 1))]

    if nIter(m.inference) == 1 # The first time data is seen, initialize all parameters
        init_onlinemodel(m, X, y)
        m.likelihood = init_likelihood(
            m.likelihood,
            m.inference,
            nLatent(m),
            size(X, 1),
            nFeatures(m),
        )
    else
        setxview!(m.inference, view(X, collect(1:nMinibatch(m.inference)), :))
        setyview!(
            m.inference,
            view_y(m.likelihood, y, collect(1:nMinibatch(m.inference))),
        )
        save_old_parameters!(m)
        m.likelihood = init_likelihood(
            m.likelihood,
            m.inference,
            nLatent(m),
            size(X, 1),
            nFeatures(m),
        )
        updateZ!(m)
    end

    # model.evol_conv = [] #Array to check on the evolution of convergence
    local_iter::Int64 = 1
    conv = Inf

    while true #loop until one condition is matched
        try #Allow for keyboard interruption without losing the model
            if local_iter == 1
                compute_old_matrices!(m)
                local_updates!(
                    m.likelihood,
                    get_y(m),
                    mean_f(m),
                    diag_cov_f(m),
                )
                ∇E_μs = ∇E_μ(m)
                ∇E_Σs = ∇E_Σ(m) # They need to be computed before recomputing the matrices
                computeMatrices!(m)
                natural_gradient!.(
                    ∇E_μs,
                    ∇E_Σs,
                    getρ(m.inference),
                    get_opt(m.inference),
                    get_Z(m),
                    m.f,
                )
                global_update!(m)
            else
                update_parameters!(m) #Update all the variational parameters
            end
            setTrained!(m, true)
            if !isnothing(callback)
                callback(m, nIter(m)) #Use a callback method if given by user
            end
            if (nIter(m.inference) % m.atfrequency == 0) &&
               nIter(m.inference) >= 3
                update_hyperparameters!(m) #Update the hyperparameters
            end
            if m.verbose > 2 || (m.verbose > 1 && local_iter % 10 == 0)
                print("Iteration : $(nIter(m.inference)) ")
                print("ELBO is : $(objective(m))")
                print("\n")
                println("number of points : $(m.f[1].dim)")
            end
            ### Print out informations about the convergence
            local_iter += 1
            m.inference.nIter += 1
            (local_iter <= iterations) || break #Verify if the number of maximum iterations has been reached
        # (iter < model.nEpochs && conv > model.ϵ) || break; #Verify if any condition has been broken
        catch e
            # if isa(e,InterruptException)
            # println("Training interrupted by user at iteration $local_iter");
            # break;
            # else
            rethrow(e)
            # end
        end
    end
    # if model.verbose > 0
    # println("Training ended after $local_iter iterations. Total number of iterations $(model.inference.nIter)")
    # end
    computeMatrices!(m) #Compute final version of the matrices for prediction
    setTrained!(m, true)
end


"""Update all variational parameters of the online sparse variational GP Model"""
function update_parameters!(model::OnlineSVGP)
    computeMatrices!(model); #Recompute the matrices if necessary (always for the stochastic case, or when hyperparameters have been updated)
    variational_updates!(model);
end


function updateZ!(model::OnlineSVGP)
    for gp in model.f
        add_point!(gp.Z,model.X,model.y,gp.kernel)
        gp.dim = gp.Z.k
    end
    model.inference.HyperParametersUpdated = true
end

function save_old_parameters!(model::OnlineSVGP)
    for gp in model.f
        save_old_gp!(gp)
    end
end

function save_old_gp!(gp::_OSVGP{T}) where {T}
    gp.Zₐ = copy(gp.Z.Z)
    remove_point!(gp.Z, kernelmatrix(gp.kernel, gp.Z, obsdim=1), gp.kernel)
    gp.invDₐ = Symmetric(-2.0*gp.η₂-inv(gp.K).mat)
    gp.prevη₁ = copy(gp.η₁)
    gp.prev𝓛ₐ = -0.5*logdet(gp.Σ) + 0.5*logdet(gp.K) - 0.5*dot(gp.μ,gp.η₁)
end

function init_onlinemodel(m::OnlineSVGP{T},X,y) where {T<:Real}
    for gp in m.f
        init_online_gp!(gp,X,y)
    end
    m.inference.xview = [view(X, collect(1:nMinibatch(m.inference)), :)]
    m.inference.yview = [view_y(m.likelihood, y, collect(1:nMinibatch(m.inference)))]
    m.inference.ρ = [1.0]
    setHPupdated!(m.inference, false)
end

function init_online_gp!(gp::_OSVGP{T}, X, y, jitt::T = T(jitt)) where {T}
    IPModule.init!(gp.Z, X, y, gp.kernel)
    nSamples = size(X, 1)
    gp.dim = gp.Z.k
    gp.Zₐ = copy(gp.Z.Z)
    gp.μ = zeros(T, gp.dim)
    gp.η₁ = zero(gp.μ)
    gp.Σ = Symmetric(Matrix(Diagonal(one(T) * I, gp.Z.k)))
    gp.η₂ = -0.5 * Symmetric(inv(gp.Σ))
    gp.K = PDMat(kernelmatrix(gp.kernel, gp.Z.Z, obsdim = 1) + jitt * I)

    gp.Kab = copy(gp.K.mat)
    gp.κₐ = Diagonal{T}(I, gp.dim)
    gp.K̃ₐ = zero(gp.Kab)

    gp.Knm = kernelmatrix(gp.kernel, X, gp.Z, obsdim = 1)
    gp.κ = gp.Knm / gp.K
    gp.K̃ =
        kerneldiagmatrix(gp.kernel, X, obsdim = 1) .+ jitt -
        opt_diag(gp.κ, gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"

    gp.invDₐ = Symmetric(Matrix{T}(I, gp.dim, gp.dim))
    gp.prev𝓛ₐ = zero(T)
    gp.prevη₁ = zero(gp.η₁)
end


function compute_old_matrices!(model::OnlineSVGP{T}) where {T}
    for gp in model.f
        compute_old_matrices!(gp, xview(model.inference), T(jitt))
    end
end

function compute_old_matrices!(gp::_OSVGP, X::AbstractMatrix, jitt::Real)
    gp.K = PDMat(kernelmatrix(gp.kernel, gp.Zₐ, obsdim = 1) + jitt * I)
    gp.Knm = kernelmatrix(gp.kernel, X, gp.Zₐ, obsdim = 1)
    gp.κ = gp.Knm / gp.K
    gp.K̃ =
        kerneldiagmatrix(gp.kernel, X, obsdim = 1) .+ jitt -
        opt_diag(gp.κ, gp.Knm)
    @assert all(gp.K̃ .> 0) "K̃ has negative values"
end
