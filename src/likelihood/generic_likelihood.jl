macro augmodel()
    @error "You need to provide a list of arguments, please check the documentation on the macro"
end

"""
** Likelihood**
Template file for likelihood creation


```julia
()
```
See all functions you need to implement
---


"""

check_model_name(name::Symbol) = !isnothing(match(r"[^\w]*",string(name)))
function check_likelihoodtype(ltype::Symbol)
    if ltype == :Regression || ltype == :Classification || ltype == :Event
        return true
    else
        return false
    end
end

function treat_likelihood(likelihood::Expr)

end

function treat_params(params)

end

function correct_parenthesis(text::AbstractString)
    replace(text,r"(?=[()*+-.])"=>"\\")
end

const AGP=AugmentedGaussianProcesses

macro augmodel(name::Symbol,likelihoodtype::Symbol,likelihood::Expr,params)
    latent = :x
    @assert occursin(r"p\(y\s*|\s*\w\)",string(likelihood)) "Likelihood should be of the form p(y|x) = C*exp(g(y)*x)*φ(α(y)-β(y)*x+γ(y)*x^2), replacing all functions by 0 if necessary"
    pdf_string = string(likelihood.args[2].args[2])
    # @show pdfexpr = Meta.parse(pdfstring)
    C_string = match(r".*?(?= (\* exp\(.*x\)))",pdf_string).match
    G_string_f = match(Regex("(?<=$(AGP.correct_parenthesis(C_string)) \\* exp\\().*(?=x\\) \\*)"),pdf_string).match
    G_string = deepcopy(G_string_f)
    while last(G_string) == ' ' || last(G_string) == '*'
        G_string = G_string[1:end-1]
    end
    phi_h_string = match(Regex("(?<=$(AGP.correct_parenthesis(G_string_f))x\\) \\* ).*"),pdf_string).match
    loc_x² = findfirst("x ^ 2",phi_h_string)
    count_parenthesis = 1
    loc_start = loc_x²[1]
    while count_parenthesis != 0
        loc_start = loc_start - 1
        if phi_h_string[loc_start] == ')'
            count_parenthesis += 1
        elseif phi_h_string[loc_start] == '('
            count_parenthesis -= 1
        end
    end
    h_string = phi_h_string[(loc_start+1):loc_x²[end]]
    phi_string = phi_h_string[1:loc_start]*"r"*phi_h_string[(loc_x²[end]+1):end]
    @show alpha_string = match(r"[^(][^-]*",h_string).match[1:end-1]
    gamma_string = match(r"(?<=\+ )[^x]*(?=x \^ 2)",h_string).match
    gamma_string = gamma_string == "" ? "1.0" : gamma_string
    while last(gamma_string) == ' ' || last(gamma_string) == '*'
        gamma_string = gamma_string[1:end-1]
    end
    beta_string = match(Regex("(?<=$(AGP.correct_parenthesis(alpha_string)) -  )[^( x )]*(?= x)"),h_string).match
    while last(beta_string) == ' ' || last(beta_string) == '*'
        beta_string = beta_string[1:end-1]
    end
    return (C_string,G_string,phi_string,alpha_string,beta_string,gamma_string)
    # treat_params(params)
    # C,g,α,β,γ,φ = treat_likelihood(likelihood)
end

macro augmodel(name::Symbol,likelihoodtype::Symbol,likelihood::Expr)

end


macro augmodel(name,ltype,C::Symbol,g::Symbol,α::Symbol,β::Symbol,γ::Symbol,φ::Symbol,∇φ::Symbol)
    #### Check args here
    @assert check_model_name(name) "Please only use alphabetic characters for the name of the likelihood"
    @assert check_likelihoodtype(ltype) "Please use a correct likelihood type : Regression, Classification or Event"
    #Find gradient with AD if needed
    #In a later stage try to find structure automatically
    esc(_augmodel(string(name),Symbol(name,"Likelihood"),Symbol(ltype,"Likelihood"),C,g,α,β,γ,φ,∇φ))
end
function _augmodel(name::String,lname,ltype,C,g,α,β,γ,φ,∇φ)
    quote begin
        # struct $(Symbol(name,"{T<:Real}"))# <: $(ltype)
        struct $(lname){T<:Real} <: AGP.$(ltype){T}
            # b::T
            c²::AGP.LatentArray{Vector{T}}
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

        function AGP.pdf(l::$(lname),y::Real,f::Real)
            $(C)()*exp($(g)(y)*f)*$(φ)($(α)(y)-$(β)(y)*f+$(γ)(y)*f^2)
        end

        function _gen_C(l::$(lname){T}) where {T}
            $(C)()
        end

        function _gen_g(l::$(lname),y::AbstractVector{T}) where {T}
            $(g).(y)
        end

        function _gen_α(l::$(lname),y::AbstractVector{T}) where {T}
            $(α).(y)
        end

        function _gen_β(l::$(lname),y::AbstractVector{T}) where {T}
            $(β).(y)
        end

        function _gen_γ(l::$(lname),y::AbstractVector{T}) where {T}
            $(γ).(y)
        end

        function _gen_φ(l::$(lname),r::T) where {T}
            $(φ).(r)
        end

        function _gen_∇φ(l::$(lname),r::T) where {T}
            $(∇φ).(r)
        end

        function _gen_∇²φ(l::$(lname),r::T) where {T}
            Zygote.gradient(x->$(∇φ)(x),r)[1]
        end

        function Base.show(io::IO,model::$(lname){T}) where {T}
            print(io,"Generic Likelihood")#WARNING TODO, to be fixed!
        end

        function Statistics.var(l::$(lname){T}) where {T}
            @warn "The variance of the likelihood is not implemented : returnin 0.0"
            return 0.0
        end

        function AGP.compute_proba(l::$(lname){T},μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
            if typeof(l) <: RegressionLikelihood
                return μ,max.(σ²,0.0).+var(l)
            elseif typeof(l) <: ClassificationLikelihood
                N = length(μ)
                pred = zeros(T,N)
                sig_pred = zeros(T,N)
                for i in 1:N
                    if σ²[i] <= 0.0
                        pred[i] = pdf(l,1.0,μ[i])
                        sig_pred[i] = 0.0
                    else
                        nodes = AGP.pred_nodes.*AGP.sqrt2.*sqrt.(σ²[i]).+μ[i]
                        pred[i] = dot(AGP.pred_weights,AGP.pdf.(l,1,nodes))
                        sig_pred[i] = dot(AGP.pred_weights,AGP.pdf.(l,1,nodes).^2) - pred[i]^2
                    end
                end
                return pred,sig_pred
            else
                @error "Prediction not implemented yet"
            end
        end

        ### Local Updates Section ###

        function AGP.local_updates!(model::VGP{T,<:$(lname),<:AnalyticVI}) where {T}
            model.likelihood.c² .= broadcast((y,μ,Σ)->_gen_α(model.likelihood,y)-_gen_β(model.likelihood,y).*μ+_gen_γ(model.likelihood,y).*(abs2.(μ)+Σ),model.inference.y,model.μ,AGP.diag.(model.Σ))
            model.likelihood.θ .= broadcast(c²->-_gen_∇φ(model.likelihood,c²)./_gen_φ(model.likelihood,c²),model.likelihood.c²)
        end

        function AGP.local_updates!(model::SVGP{T,<:$(lname),<:AnalyticVI}) where {T}
            model.likelihood.c² .= broadcast((y,κ,μ,Σ,K̃)->_gen_α(model.likelihood,y)-_gen_β(model.likelihood,y).*(κ*μ)+_gen_γ(model.likelihood,y).*(abs2.(κ*μ)+AGP.opt_diag(κ*Σ,κ)+K̃),model.inference.y,model.κ,model.μ,model.Σ,model.K̃)
            model.likelihood.θ .= broadcast(c²->-_gen_∇φ.(model.likelihood,c²)./_gen_φ.(model.likelihood,c²),model.likelihood.c²)
        end

        function sample_omega(::$(lname),f)
            @error "You cannot use Gibbs sampling from your likelihood unless you define sample_omega(likelihood,f)"
        end

        function AGP.sample_local!(model::VGP{T,<:$(lname),<:GibbsSampling}) where {T}
            model.likelihood.θ .= broadcast((y,μ)->pω.(model.likelihood,sqrt.(α(model.likelihood,model.likelihood,y)-_gen_β(model.likelihood,y).*μ+_gen_γ(model.likelihood,y).*(μ.^2))),model.inference.y,model.μ)
        end

        ### Natural Gradient Section ###

        @inline AGP.∇E_μ(model::AbstractGP{T,<:$(lname),<:AGP.GibbsorVI}) where {T} = broadcast((y,θ)->_gen_g(model.likelihood,y)+θ.*_gen_β(model.likelihood,y),model.inference.y,model.likelihood.θ)
        @inline AGP.∇E_μ(model::AbstractGP{T,<:$(lname),<:AGP.GibbsorVI},i::Int) where {T} =  _gen_g(model.likelihood,model.inference.y[i])+model.likelihood.θ[i].*_gen_β(model.likelihood,model.inference.y[i])
        @inline AGP.∇E_Σ(model::AbstractGP{T,<:$(lname),<:AGP.GibbsorVI}) where {T} =
        broadcast((y,θ)->θ.*_gen_γ(model.likelihood,y),model.inference.y,model.likelihood.θ)
        @inline AGP.∇E_Σ(model::AbstractGP{T,<:$(lname),<:AGP.GibbsorVI},i::Int) where {T} = model.likelihood.θ[i].*_gen_γ(model.likelihood,model.inference.y[i])

        ### ELBO Section ###

        function AGP.ELBO(model::AbstractGP{T,<:$(lname),<:AnalyticVI}) where {T}
            return AGP.expecLogLikelihood(model) - AGP.GaussianKL(model) - AugmentedKL(model)
        end

        function AGP.expecLogLikelihood(model::VGP{T,<:$(lname),<:AnalyticVI}) where {T}
            tot = model.nLatent*model.nSample*log(_gen_C(model.likelihood))
            tot += sum(broadcast((y,μ)->dot(_gen_g(model.likelihood,y),μ),model.inference.y,model.μ))
            tot += -sum(broadcast((θ,y,μ,Σ)-> dot(θ,_gen_α(model.likelihood,y))
                                            - dot(θ,_gen_β(model.likelihood,y).*μ)
                                            + dot(θ,_gen_γ(model.likelihood,y).*(abs2.(μ)+Σ)),
                                            model.likelihood.θ,model.inference.y,
                                            model.μ,diag.(model.Σ)))
            return tot
        end

        function AGP.expecLogLikelihood(model::SVGP{T,<:$(lname),<:AnalyticVI}) where {T}
            tot = model.nLatent*model.inference.nSamplesUsed*log(_gen_C(model.likelihood))
            tot += sum(broadcast((y,κμ)->dot(_gen_g(model.likelihood,y),κμ),model.inference.y,model.κ.*model.μ))
            tot += -sum(broadcast((θ,y,κμ,κΣκ,K̃)->dot(θ,_gen_α(model.likelihood,y))
                                            - dot(θ,_gen_β(model.likelihood,y).*κμ)
                                            + dot(θ,_gen_γ(model.likelihood,y).*(abs2.(κμ)+κΣκ+K̃)),
                                            model.likelihood.θ,model.inference.y,
                                            model.κ.*model.μ,opt_diag.(model.κ.*model.Σ,model.κ),model.K̃))
            return model.inference.ρ*tot
        end

        function AugmentedKL(model::AbstractGP{T,<:$(lname),<:AnalyticVI}) where {T}
            # AGP.local_updates!(model)
            model.inference.ρ*sum(broadcast((c²,θ)->-dot(c²,θ)-sum(log,_gen_φ.(model.likelihood,c²)),model.likelihood.c²,model.likelihood.θ))
        end

        ### Gradient Section ###

        @inline function AGP.grad_log_pdf(l::$(lname){T},y::Real,f::Real) where {T<:Real}
            h² = _gen_α(y) - _gen_β(y)*f + _gen_γ(y)*f^2
            _gen_g(y)+(-_gen_β(y)+2*_gen_γ(y)*f)*_gen_∇φ(h²)/_gen_φ(h²)
        end

        @inline function AGP.hessian_log_pdf(l::$(lname){T},y::Real,f::Real) where {T<:Real}
            h² = _gen_α(y) - _gen_β(y)*f + _gen_γ(y)*f^2
            φ = _gen_φ(l,h²); ∇φ = _gen_∇φ(l,h²); ∇²φ = _gen_∇²φ(l,h²)
            return (2*_gen_γ(y)*∇φ/φ
                    -((-_gen_β(y)+2*_gen_γ(y)*f)*∇φ/φ)^2
                    +(-_gen_β(y)+2*_gen_γ(y)*f)^2*∇²φ/φ)
        end

    end
end
end
