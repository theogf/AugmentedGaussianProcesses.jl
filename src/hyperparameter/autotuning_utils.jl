
### Global constant allowing to chose between forward_diff and reverse_diff for hyperparameter optimization ###
const ADBACKEND = Ref(:reverse_diff)

const Z_ADBACKEND = Ref(:auto)

const K_ADBACKEND = Ref(:auto)

function setadbackend(backend_sym)
    @assert backend_sym == :forward_diff || backend_sym == :reverse_diff
    ADBACKEND[] = backend_sym
end

function setKadbackend(backend_sym)
    @assert backend_sym == :forward_diff || backend_sym == :reverse_diff || backend_sym == :auto
    K_ADBACKEND[] = backend_sym
end

function setZadbackend(backend_sym)
    @assert backend_sym == :forward_diff || backend_sym == :reverse_diff || backend_sym == :auto
    Z_ADBACKEND[] = backend_sym
end

##
function apply_grads_kernel_params!(opt, k::Kernel, Δ::IdDict)
    ps = Flux.params(k)
    for p in ps
        Δ[p] == nothing && continue
        Δlogp = Flux.Optimise.apply!(opt, p, p .* vec(Δ[p]))
        p .= exp.(log.(p) + Δlogp)
    end
end

function apply_gradients_mean_prior!(μ::PriorMean,g::AbstractVector,X::AbstractMatrix)
    update!(μ,g,X)
end
