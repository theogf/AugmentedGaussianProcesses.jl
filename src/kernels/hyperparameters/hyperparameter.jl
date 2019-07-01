#== HyperParameter Type ==#

mutable struct HyperParameter{T<:Real}
    value::T
    interval::Interval{T}
    fixed::Bool
    opt::Optimizer
    # function HyperParameter{T}(x::T,I::Interval{T};fixed::Bool=false,opt::Optimizer=Momentum(η=0.01)) where {T<:Real}
    function HyperParameter{T}(x::T,I::Interval{T};fixed::Bool=false,opt::Optimizer=Adam(α=0.01)) where {T<:Real}
    # function HyperParameter{T}(x::T,I::Interval{T};fixed::Bool=false,opt::Optimizer=VanillaGradDescent(η=0.001)) where {T<:Real}
        checkvalue(I, x) || error("Value $(x) must be in range " * string(I))
        new{T}(x, I, fixed, opt)
    end
end
HyperParameter(x::T, I::Interval{T} = interval(T); fixed::Bool = false, opt::Optimizer = Adam(α=0.01)) where {T<:Real} = HyperParameter{T}(x, I, fixed, opt)

eltype(::HyperParameter{T}) where {T} = T

@inline getvalue(θ::HyperParameter{T}) where {T} = θ.value

function setvalue!(θ::HyperParameter{T}, x::T) where {T}
    checkvalue(θ.interval, x) || error("Value $(x) must be in range " * string(θ.interval))
    θ.value = x
    return θ
end

function setparamoptimizer!(θ::HyperParameter{T},opt::Optimizer) where {T}
    θ.opt = copy(opt)
end

checkvalue(θ::HyperParameter{T}, x::T) where {T} = checkvalue(θ.interval, x)

convert(::Type{HyperParameter{T}}, θ::HyperParameter{T}) where {T<:Real} = θ
function convert(::Type{HyperParameter{T}}, θ::HyperParameter) where {T<:Real}
    HyperParameter{T}(convert(T, getvalue(θ)), convert(Interval{T}, θ.bounds))
end
convert(::Type{T}, θ::HyperParameter{T}) where {T<:Number} = T(θ.value)

function show(io::IO, θ::HyperParameter{T}) where {T}
    print(io, string("HyperParameter(", getvalue(θ), ",", string(θ.interval), ")"))
end

gettheta(θ::HyperParameter) = theta(θ.interval, getvalue(θ))

settheta!(θ::HyperParameter, x::T) where {T}= setvalue!(θ, eta(θ.interval,x))

checktheta(θ::HyperParameter, x::T) where {T} = checktheta(θ.interval, x)

getderiv_eta(θ::HyperParameter) = deriv_eta(θ.interval, getvalue(θ))

for op in (:isless, :(==), :+, :-, :*, :/)
    @eval begin
        $op(θ1::HyperParameter, θ2::HyperParameter) = $op(getvalue(θ1), getvalue(θ2))
        $op(a::Number, θ::HyperParameter) = $op(a, getvalue(θ))
        $op(θ::HyperParameter, a::Number) = $op(getvalue(θ), a)
    end
end
###Old version not using reparametrization
update!(param::HyperParameter{T},grad::T) where {T} = begin
    # println("Correc : $(getderiv_eta(param)), Grad : $(GradDescent.update(param.opt,grad)), theta : $(gettheta(param))")
    isfree(param) ? settheta!(param, gettheta(param) + update(param.opt,getderiv_eta(param)*grad)) : nothing
    # isfree(param) ? setvalue!(param, getvalue(param) + GradDescent.update(param.opt,grad)) : nothing
end

isfree(θ::HyperParameter) = !θ.fixed

setfixed!(θ::HyperParameter) = θ.fixed = true

setfree!(θ::HyperParameter) = θ.fixed = false

mutable struct HyperParameters{T<:Real}
    hyperparameters::Array{HyperParameter{T},1}
    function HyperParameters{T}(θ::Vector{T},intervals::Array{Interval{T,A,B}}) where {A<:Bound{T},B<:Bound{T}} where {T<:Real}
        this = new(Vector{HyperParameter{T}}(undef,length(θ)))
        for i in 1:length(θ)
            this.hyperparameters[i] = HyperParameter{T}(θ[i],intervals[i])
        end
        return this
    end
end

function HyperParameters(θ::Vector{T},intervals::Vector{Interval{T,A,B}}) where {A<:Bound{T},B<:Bound{T}} where {T<:Real}
    HyperParameters{T}(θ,intervals)
end

function setparamoptimizer!(θs::HyperParameters{T},opt::Optimizer) where {T}
    for θ in θs.hyperparameters
        θ.opt = copy(opt)
    end
end

@inline getvalue(θ::HyperParameters{T}) where T = broadcast(getvalue,θ.hyperparameters)

function Base.getindex(p::HyperParameters{T},it::Integer) where T
    return p.hyperparameters[it]
end

function update!(param::HyperParameters{T},grad::Vector{T}) where T
    update!.(param.hyperparameters,grad)
end

setfixed!(θ::HyperParameters{T}) where T = setfixed!.(θ.hyperparameters)

setfree!(θ::HyperParameters{T}) where T = setfree!.(θ.hyperparameters)
