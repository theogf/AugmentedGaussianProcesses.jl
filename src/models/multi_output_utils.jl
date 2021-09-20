##
@traitfn function mean_f(model::TGP, state) where {T,TGP<:AbstractGPModel{T};IsMultiOutput{TGP}}
    μ_q = mean_f.(model.f, state.kernel_matrices)
    μ_f = []
    for i in 1:(model.nTask)
        x = ntuple(model.nf_per_task[i]) do _
            zeros(T, batchsize(inference(model)))
        end
        for j in 1:model.nf_per_task[i]
            x[j] .= sum(model.A[i][j] .* μ_q)
        end
        push!(μ_f, x)
    end
    return μ_f
end

##
@traitfn function var_f(model::TGP, state) where {T,TGP<:AbstractGPModel{T};IsMultiOutput{TGP}}
    Σ_q = var_f.(model.f, state.kernel_matrices)
    Σ_f = []
    for i in 1:(model.nTask)
        x = ntuple(model.nf_per_task[i]) do _
            zeros(T, batchsize(inference(model)))
        end
        for j in 1:model.nf_per_task[i]
            x[j] .= sum(model.A[i][j] .^ 2 .* Σ_q)
        end
        push!(Σ_f, x)
    end
    return Σ_f
end

## return the linear sum of the expectation gradient given μ ##
@traitfn function ∇E_μ(m::TGP, ys, state) where {T,TGP<:AbstractGPModel{T};IsMultiOutput{TGP}}
    ∇ = [zeros(T, batchsize(inference(m))) for i in 1:n_latent(m)]
    ∇Eμs = ∇E_μ.(likelihood(m), Ref(opt(inference(m))), ys, state.local_vars)
    ∇EΣs = ∇E_Σ.(likelihood(m), Ref(opt(inference(m))), ys, state.local_vars)
    μ_f = mean_f.(m.f, state.kernel_matrices)
    for t in 1:(m.nTask)
        for j in 1:m.nf_per_task[t]
            for q in 1:n_latent(m)
                ∇[q] .+=
                    m.A[t][j][q] * (
                        ∇Eμs[t][j] -
                        2 * ∇EΣs[t][j] .*
                        sum(m.A[t][j][qq] * μ_f[qq] for qq in 1:n_latent(m) if qq != q)
                    )
            end
        end
    end
    return ∇
end

## return the linear sum of the expectation gradient given diag(Σ) ##
@traitfn function ∇E_Σ(m::TGP, ys, state) where {T,TGP<:AbstractGPModel{T};IsMultiOutput{TGP}}
    ∇ = [zeros(T, batchsize(inference(m))) for _ in 1:n_latent(m)]
    ∇Es = ∇E_Σ.(likelihood(m), Ref(opt(inference(m))), ys, state.local_vars)
    for t in 1:(m.nTask)
        for j in 1:m.nf_per_task[t]
            for q in 1:n_latent(m)
                ∇[q] .+= m.A[t][j][q]^2 * ∇Es[t][j]
            end
        end
    end
    return ∇
end

##
@traitfn function update_A!(m::TGP, ys, state) where {T,TGP<:AbstractGPModel{T};IsMultiOutput{TGP}}
    if !isnothing(m.A_opt)
        local_vars = state.local_vars
        μ_f = mean_f.(m.f, state.kernel_matrices) # κμ || μ
        Σ_f = var_f.(m.f, state.kernel_matrices) # Diag(K̃ + κΣκ) || Diag(Σ)
        ∇Eμ = ∇E_μ.(likelihood(m), Ref(opt(inference(m))), ys, local_vars)
        ∇EΣ = ∇E_Σ.(likelihood(m), Ref(opt(inference(m))), ys, local_vars)
        opt_state = state.opt_state
        # new_A = zero(model.A)
        for t in 1:(m.nTask)
            for j in 1:m.nf_per_task[t]
                ∇A = zero(m.A[t][j])
                for q in 1:n_latent(m)
                    x1 =
                        dot(∇Eμ[t][j], μ_f[q]) -
                        2 * dot(
                            ∇EΣ[t][j],
                            μ_f[q] .*
                            sum(m.A[t][j][qq] * μ_f[qq] for qq in 1:n_latent(m) if qq != q),
                        )
                    x2 = dot(∇EΣ[t][j], abs2.(μ_f[q]) + Σ_f[q])
                    # new_A[t,j,q] = x1/(2*x2)
                    ∇A[q] = x1 - 2 * m.A[t][j][q] * x2
                end
                m.A[t][j] .+= Optimiserse.apply(m.A_opt, m.A[t][j], ∇A)
                m.A[t][j] /= sqrt(sum(abs2, m.A[t][j])) # Projection on the unit circle
            end
        end
    end
end
