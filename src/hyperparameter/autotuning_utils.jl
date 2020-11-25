
### Global constant allowing to chose between forward_diff and zygote_diff for hyperparameter optimization ###
const ADBACKEND = Ref(:zygote)

const Z_ADBACKEND = Ref(:auto)

const K_ADBACKEND = Ref(:auto)

function setadbackend(ad_backend::Symbol)
    (ad_backend == :forward || ad_backend == :zygote) ||
        error("Wrong backend symbol, options are :forward or :zygote")
    ADBACKEND[] = ad_backend
end

function setKadbackend(ad_backend::Symbol)
    (ad_backend == :forward || ad_backend == :zygote || ad_backend == :auto) ||
        error("Wrong backend symbol, options are :forward, :zygote or :auto")
    K_ADBACKEND[] = ad_backend
end

function setZadbackend(ad_backend::Symbol)
    (ad_backend == :forward || ad_backend == :zygote || ad_backend == :auto) ||
        error("Wrong backend symbol, options are :forward, :zygote or :auto")
    Z_ADBACKEND[] = ad_backend
end

## Updating kernel parameters ##
function apply_Δk!(opt, k::Kernel, Δ::IdDict)
    ps = params(k)
    for p in ps
        isnothing(Δ[p]) && continue
        Δlogp = Optimise.apply!(opt, p, p .* vec(Δ[p]))
        p .= exp.(log.(p) + Δlogp)
    end
end

function apply_Δk!(opt, k::Kernel, Δ::AbstractVector)
    ps = params(k)
    i = 1
    for p in ps
        d = length(p)
        Δp = Δ[i:i+d-1]
        Δlogp = Optimise.apply!(opt, p, p .* Δp)
        @. p = exp(log(p) + Δlogp)
        i += d
    end
end


function apply_gradients_mean_prior!(μ::PriorMean, g::AbstractVector, X::AbstractVector)
    update!(μ, g, X)
end

function update!(opt, Z::AbstractInducingPoints, Z_grads)
    for (z, zgrad) in zip(Z, Z_grads)
        z .+= Optimise.apply!(opt, z, zgrad)
    end
end
