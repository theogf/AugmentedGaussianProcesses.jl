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
const InputSymbol = Union{Symbol,Expr,Real}


check_model_name(name::Symbol) = !isnothing(match(r"[^\w]*",string(name)))

check_likelihoodtype(ltype::Symbol) = Symbol(ltype) == :Regression || Symbol(ltype) == :Classification || Symbol(ltype) == :Event

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

macro augmodel(name,ltype,C::InputSymbol,g::InputSymbol,α::InputSymbol,β::InputSymbol,γ::InputSymbol,φ::InputSymbol,∇φ::InputSymbol,args...)
    add_variables = []; default_values = [];
    for input in args
        @assert input.head == :(=) "Additional variables should be given a default value, for example `b=1`"
        push!(add_variables,input.args[1])
        push!(default_values,input.args[1])
    end
    #### Check args here
    @assert check_model_name(name) "Please only use alphabetic characters for the name of the likelihood"
    @assert check_likelihoodtype(ltype) "Please use a correct likelihood type : Regression, Classification or Event"
    #Find gradient with AD if needed
    esc(_augmodel(string(name),name,Symbol(ltype,"Likelihood"),C,g,α,β,γ,φ,∇φ))
end
function _augmodel(name::String,lname,ltype,C,g,α,β,γ,φ,∇φ)
    quote begin
        using Statistics
        # struct $(Symbol(name,"{T<:Real}"))# <: $(ltype)
        struct $(lname){T<:Real} <: AGP.$(ltype){T}
            # b::T
            c²::AGP.LatentArray{Vector{T}}
            θ::AGP.LatentArray{Vector{T}}
            function $(lname){T}() where {T<:Real}
                new{T}()
            end
            function $(lname){T}($(add_variables) c²::AbstractVector{<:AbstractVector{<:Real}},θ::AbstractVector{<:AbstractVector{<:Real}}) where {T<:Real}
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
            C(l)*exp(g(l,y)*f)*φ(l,α(l,y)-β(l,y)*f+γ(l,y)*f^2)
        end

        function AGP.logpdf(l::$(lname),y::Real,f::Real)
            log(C(l))+g(l,y)*f+log(φ(l,α(l,y)-β(l,y)*f+γ(l,y)*f^2))
        end

        function C(l::$(lname){T}) where {T}
            C()
        end

        function _gen_g(l::$(lname),y::AbstractVector{T}) where {T}
            $(g)
        end

        function _gen_α(l::$(lname),y::AbstractVector{T}) where {T}
            $(α)
        end

        function _gen_β(l::$(lname),y::AbstractVector{T}) where {T}
            $(β)
        end

        function _gen_γ(l::$(lname),y::AbstractVector{T}) where {T}
            $(γ)
        end

        function _gen_φ(l::$(lname),r::T) where {T}
            $(φ)
        end

        function _gen_∇φ(l::$(lname),r::T) where {T}
            $(∇φ)
        end

        function _gen_∇²φ(l::$(lname),r::T) where {T}
            Zygote.gradient(x->$(∇φ)(x),r)[1]
        end

        function Base.show(io::IO,model::$(lname){T}) where {T}
            print(io,"$(:($lname))")
        end

        function Statistics.var(l::$(lname){T}) where {T}
            @warn "The variance of the likelihood is not implemented : returnin 0.0"
            return 0.0
        end

        function AGP.compute_proba(l::$(lname){T},y::AbstractVector,μ::AbstractVector{T},σ²::AbstractVector{T}) where {T<:Real}
            if typeof(l) <: RegressionLikelihood
                return μ,max.(σ²,zero(T)).+var(l)
            elseif typeof(l) <: ClassificationLikelihood
                N = length(μ)
                pred = zeros(T,N)
                sig_pred = zeros(T,N)
                for i in 1:N
                    x = AGP.pred_nodes.*sqrt(max(σ²[i],zero(T))).+μ[i]
                    pred[i] = dot(AGP.pred_weights,AGP.pdf.(l,1,x))
                    sig_pred[i] = dot(AGP.pred_weights,AGP.pdf.(l,1,x).^2) - pred[i]^2
                end
                return pred,sig_pred
            else
                @error "Prediction not implemented yet, please file an issue"
            end
        end

        ### Local Updates Section ###

        function AGP.local_updates!(l::$(lname),y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
            l.c² .= α(l,y)-β(l,y).*μ+γ(l,y).*(abs2.(μ)+diag_cov)
            l.θ .= -∇φ(l,l.c²)./φ(l,l.c²)
        end

        function pω(::$(lname),f)
            #TODO Use the Laplace approx sampler
            @error "You cannot use Gibbs sampling from your likelihood unless you define pω(likelihood,f)"
        end

        function AGP.sample_local!(l::$(lname),y::AbstractVector,f::AbstractVector) where {T}
            set_ω!(l,pω.(l,sqrt.(0.5*(l,α(l,y)-β(l,y).*f+γ(l,y).*(abs2.(f))))))
        end

        ### Natural Gradient Section ###

        @inline AGP.∇E_μ(l::$(lname),::AOptimizer,y::AbstractVector) where {T} = g(l,y)+l.θ.*β(l,y)
        @inline AGP.∇E_Σ(l::$(lname),::AOptimizer,y::AbstractVector) where {T} = l.θ.*γ(l,y)

        ### ELBO Section ###
        function AGP.expec_logpdf(l::$(lname),i::AnalyticVI,y::AbstractVector,μ::AbstractVector,diag_cov::AbstractVector) where {T}
            tot = length(y)*log(C(l))
            tot += dot(g(l,y),μ)
            tot += -(dot(θ,α(l,y))
                     - dot(θ,β(l,y).*μ)
                     + dot(θ,γ(l,y).*(abs2.(μ)+diag_cov)))
            return tot
        end

        function AugmentedKL(model::AbstractGP{T,<:$(lname),<:AnalyticVI}) where {T}
            -dot(l.c²,l.θ)-sum(log,φ.(l,c²))
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
