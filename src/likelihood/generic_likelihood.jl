macro augmodel()
    @error "You need to provide a list of arguments, please check the documentation on the macro"
end

"""
**$(local sname) Likelihood**

Template file for likelihood creation

```julia
$(local lname)()
```
See all functions you need to implement
---


"""
const AGP=AugmentedGaussianProcesses
macro augmodel(name::String,LikelihoodType::String,C::Symbol,g::Symbol,α::Symbol,β::Symbol,γ::Symbol,φ::Symbol,∇φ::Symbol)
    #### Check args here
    #Check name has no space
    #Check LikelihoodType exists
    #Find gradient with AD if needed
    #In a later stage try to find structure automatically
    esc(_augmodel(Symbol(name),Symbol(name,"Likelihood"),Symbol(name,"Likelihood{T}"),Symbol(LikelihoodType*"Likelihood"),C,g,α,β,γ,φ,∇φ))
end
function _augmodel(name,lname,lnameT,ltype,C,g,α,β,γ,φ,∇φ)
    quote begin
        # struct $(Symbol(name,"{T<:Real}"))# <: $(ltype)
        struct $(lname){T<:Real} <: AGP.$(ltype){T}
            # b::T
            # c²::AGP.LatentArray{Vector}
            c²::AGP.LatentArray{Vector{T}}
            # θ::AGP.LatentArray{Vector}
            θ::AGP.LatentArray{Vector{T}}
            function $(lname){T}() where {T<:Real}
                new{T}()
            end
            function $(lname){T}(c²::AbstractVector{<:AbstractVector{<:Real}},θ::AbstractVector{<:AbstractVector{<:Real}}) where {T<:Real}
                new{T}(c²,θ)
            end
        end

        function $(lname)()
            $(lname){Float64}()
        end

        function AGP.init_likelihood(likelihood::$(lname){T},inference::Inference{T},nLatent::Int,nSamplesUsed::Int,nFeatures::Int) where T
            if inference isa AnalyticVI || inference isa GibbsSampling
                $(lname){T}([zeros(T,nSamplesUsed) for _ in 1:nLatent],[zeros(T,nSamplesUsed) for _ in 1:nLatent])
            else
                $(lname){T}()
            end
        end

        function C(l::$(lname){T}) where {T}
            C()
        end

        function g(l::$(lname),y::AbstractVector{T}) where {T}
            g.(y)
        end

        function α(l::$(lname),y::AbstractVector{T}) where {T}
            α.(y)
        end

        function β(l::$(lname),y::AbstractVector{T}) where {T}
            β.(y)
        end

        function γ(l::$(lname),y::AbstractVector{T}) where {T}
            γ.(y)
        end

        function φ(l::$(lname),r::T) where {T}
            φ.(r)
        end

        function ∇φ(l::$(lname),r::T) where {T}
            ∇φ.(r)
        end

        function pdf(l::$(lname),y::Real,f::Real)
            C()*exp(g(y)*f)*φ(α(y)-β(y)*f-α(y)*f)
        end

        function Base.show(io::IO,model::$(lname){T}) where {T}
            print(io,"$(name) Likelihood")
        end

        function AGP.compute_proba(l::$(lname){T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
            if typeof(l) <: RegressionLikelihood
                return μ,max.(σ²,0.0).+ 1.0/(C())^2 #TODO Mamene
            elseif typeof(l) <: ClassificationLikelihood
                N = length(μ)
                pred = zeros(T,N)
                for i in 1:N
                    if σ²[i] <= 0.0
                        pred[i] = pdf(likelihood,μ[i])
                    else
                        nodes = pred_nodes.*sqrt2.*sqrt.(σ²[i]).+μ[i]
                        pred[i] = dot(pred_weights,logistic.(nodes))
                    end
                end
                return pred
            else
                @error "Prediction not implemented yet"
            end
        end

        ### Local Updates Section ###

        function AGP.local_updates!(model::VGP{T,<:$(lname),<:AnalyticVI}) where {T}
            model.likelihood.c² .= broadcast((y,μ,Σ)->α(model.likelihood,y)-β(model.likelihood,y).*μ+γ(model.likelihood,y).*(abs2.(μ)+Σ),model.inference.y,model.μ,AGP.diag.(model.Σ))
            model.likelihood.θ .= broadcast(c²->-∇φ(model.likelihood,c²)./φ(model.likelihood,c²),model.likelihood.c²)
        end

        function AGP.local_updates!(model::SVGP{T,<:$(lname),<:AnalyticVI}) where {T}
            model.likelihood.c² .= broadcast((y,κ,μ,Σ)->α.(model.likelihood,y)-β.(model.likelihood,y).*(κ*μ)+γ.(model.likelihood,y).*(abs2.(κ*μ)+AGP.opt_diag(κ*Σ,κ)),model.inference.y,model.κ,model.μ,model.Σ)
            model.likelihood.θ .= broadcast(c²->-∇φ.(model.likelihood,c²)./φ.(model.likelihood,c²),model.likelihood.c²)
        end

        function AGP.sample_local!(model::VGP{T,<:$(lname),<:GibbsSampling}) where {T}
            return nothing
        end

        ### Natural Gradient Section ###

        @inline AGP.∇E_μ(model::AbstractGP{T,<:$(lname),<:AGP.GibbsorVI}) where {T} = broadcast((y,θ)->g(model.likelihood,y)+θ.*β(model.likelihood,y),model.inference.y,model.likelihood.θ)
        @inline AGP.∇E_μ(model::AbstractGP{T,<:$(lname),<:AGP.GibbsorVI},i::Int) where {T} =  g(model.likelihood,model.inference.y[i])+model.likelihood.θ[i].*β(model.likelihood,model.inference.y[i])
        @inline AGP.∇E_Σ(model::AbstractGP{T,<:$(lname),<:AGP.GibbsorVI}) where {T} =
        broadcast((y,θ)->θ.*γ(model.likelihood,y),model.inference.y,model.likelihood.θ)
        @inline AGP.∇E_Σ(model::AbstractGP{T,<:$(lname),<:AGP.GibbsorVI},i::Int) where {T} = model.likelihood.θ[i].*γ(model.likelihood,model.inference.y[i])

        ### ELBO Section ###

        function AGP.ELBO(model::AbstractGP{T,<:$(lname),<:AnalyticVI}) where {T}
            return expecLogLikelihood(model) - GaussianKL(model) - AugmentedKL(model)
        end

        function AGP.expecLogLikelihood(model::VGP{T,<:$(lname),<:AnalyticVI}) where {T}
            tot = model.nLatent*model.nSamples*log(C(model.likelihood))
            tot += sum(broadcast((y,μ)->dot(g(model.likelihood,y),μ),model.inference.y,model.μ))
            tot += -sum(broadcast((θ,y,μ,Σ)->dot(θ,α(model.likelihood,y))
                                            - dot(θ,β(model.likelihood,y).*μ)
                                            + dot(θ,γ(model.likelihood,y).*(abs2.(μ)+Σ)),
                                            model.likelihood.θ,model.inference.y,
                                            model.μ,diag.(model.Σ)))
            return tot
        end

        function AGP.expecLogLikelihood(model::SVGP{T,<:$(lname),<:AnalyticVI}) where {T}
            tot = 0.0
            return model.inference.ρ*tot
        end

        function AugmentedKL(model::AbstractGP{T,<:$(lname),<:AnalyticVI}) where {T}
            model.inference.ρ*sum(broadcast((c²,θ)->-sum(c².*θ)-sum(log,φ.(model.likelihood,c²)),model.likelihood.c²,model.likelihood,θ))
        end

        ### Gradient Section ###

        function gradpdf(::$(lname),y::Int,f::T) where {T<:Real}
        end

        function hessiandiagpdf(::$(lname),y::Int,f::T) where {T<:Real}
        end

    end
end
end
