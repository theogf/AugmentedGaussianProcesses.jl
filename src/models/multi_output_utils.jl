@traitfn get_y(model::TGP) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}} = view_y.(model.likelihood, model.y, model.inference.MBIndices)

##
@traitfn function mean_f(model::TGP) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}}
    μ_q = mean_f.(model.f)
    μ_f = []
    for i in 1:model.nTask
        x = ntuple(_->zeros(T,model.inference.nMinibatch),model.nf_per_task[i])
        for j in 1:model.nf_per_task[i]
            x[j] .= sum(model.A[i][j] .* μ_q)
        end
        push!(μ_f,x)
    end
    return μ_f
end

##
@traitfn function diag_cov_f(model::TGP) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}}
    Σ_q = diag_cov_f.(model.f)
    Σ_f = []
    for i in 1:model.nTask
        x = ntuple(_->zeros(T,model.inference.nMinibatch),model.nf_per_task[i])
        for j in 1:model.nf_per_task[i]
            x[j] .= sum(model.A[i][j].^2 .* Σ_q)
        end
        push!(Σ_f,x)
    end
    return Σ_f
end


## return the linear sum of the expectation gradient given μ ##
@traitfn function ∇E_μ(m::TGP) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}}
    ∇ = [zeros(T,m.inference.nMinibatch[mod(i, nX(m))+1]) for i in 1:nLatent(m)]
    ∇Eμs = ∇E_μ.(m.likelihood,m.inference.vi_opt[1:1],get_y(m))
    ∇EΣs = ∇E_Σ.(m.likelihood,m.inference.vi_opt[1:1],get_y(m))
    μ_f = mean_f.(m.f)
    for t in 1:m.nTask
        for j in 1:m.nf_per_task[t]
            for q in 1:nLatent(m)
                ∇[q] .+= m.A[t][j][q] * (∇Eμs[t][j]  - 2*∇EΣs[t][j].*sum(m.A[t][j][qq]*μ_f[qq] for qq in 1:nLatent(m) if qq != q))
            end
        end
    end
    return ∇
end

## return the linear sum of the expectation gradient given diag(Σ) ##
@traitfn function ∇E_Σ(m::TGP) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}}
    ∇ = [zeros(T,m.inference.nMinibatch[mod(i, nX(m))+1]) for i in 1:m.nLatent]
    ∇Es = ∇E_Σ.(m.likelihood,m.inference.vi_opt[1:1],get_y(m))
    for t in 1:m.nTask
        for j in 1:m.nf_per_task[t]
            for q in 1:nLatent(m)
                ∇[q] .+= m.A[t][j][q]^2 * ∇Es[t][j]
            end
        end
    end
    return ∇
end

##
@traitfn function update_A!(model::TGP) where {T,TGP<:AbstractGP{T};IsMultiOutput{TGP}}
    if !isnothing(model.A_opt)
        μ_f = mean_f.(model.f) # κμ || μ
        Σ_f = diag_cov_f.(model.f) #Diag(K̃ + κΣκ) || Diag(Σ)
        ∇Eμ = ∇E_μ.(model.likelihood, model.inference.vi_opt[1:1], get_y(model))
        ∇EΣ = ∇E_Σ.(model.likelihood, model.inference.vi_opt[1:1], get_y(model))
        # new_A = zero(model.A)
        for t in 1:model.nTask
            for j in 1:model.nf_per_task[t]
                ∇A = zero(model.A[t][j])
                for q in 1:model.nLatent
                    x1 = dot(∇Eμ[t][j], μ_f[q]) - 2 * dot(∇EΣ[t][j], μ_f[q] .* sum(model.A[t][j][qq] * μ_f[qq] for qq in 1:model.nLatent if qq!=q))
                    x2 = dot(∇EΣ[t][j], abs2.(μ_f[q]) + Σ_f[q])
                    # new_A[t,j,q] = x1/(2*x2)
                    ∇A[q] = x1 - 2 * model.A[t][j][q]*  x2
                end
                model.A[t][j] .+= Flux.Optimise.apply!(model.A_opt, model.A[t][j], ∇A)
                model.A[t][j] /= sqrt(sum(abs2,model.A[t][j])) # Projection on the unit circle
            end
        end
    end
end
