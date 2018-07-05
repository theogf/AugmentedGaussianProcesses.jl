#== HyperParameter Type ==#

mutable struct HyperParameter{T<:Real}
    value::Base.RefValue{T}
    interval::Interval{T}
    opt::Optimizer
    fixed::Bool
    # function HyperParameter{T}(x::T,I::Interval{T};fixed::Bool=false,opt::Optimizer=Momentum(η=0.5)) where {T<:Real}
    function HyperParameter{T}(x::T,I::Interval{T};fixed::Bool=false,opt::Optimizer=Adam(α=0.01)) where {T<:Real}
        checkvalue(I, x) || error("Value $(x) must be in range " * string(I))
        new{T}(Ref(x), I, opt, fixed)
    end
end
HyperParameter{T<:Real}(x::T, I::Interval{T} = interval(T); fixed::Bool = false, opt::Optimizer = Adam(α=0.01)) = HyperParameter{T}(x, I, fixed, opt)

eltype{T}(::HyperParameter{T}) = T

@inline getvalue{T}(θ::HyperParameter{T}) = getindex(θ.value)

function setvalue!{T}(θ::HyperParameter{T}, x::T)
    checkvalue(θ.interval, x) || error("Value $(x) must be in range " * string(θ.interval))
    setindex!(θ.value, x)
    return θ
end

checkvalue{T}(θ::HyperParameter{T}, x::T) = checkvalue(θ.interval, x)

convert{T<:Real}(::Type{HyperParameter{T}}, θ::HyperParameter{T}) = θ
function convert{T<:Real}(::Type{HyperParameter{T}}, θ::HyperParameter)
    HyperParameter{T}(convert(T, getvalue(θ)), convert(Interval{T}, θ.bounds))
end

function show{T}(io::IO, θ::HyperParameter{T})
    print(io, string("HyperParameter(", getvalue(θ), ",", string(θ.interval), ")"))
end

gettheta(θ::HyperParameter) = theta(θ.interval, getvalue(θ))

settheta!{T}(θ::HyperParameter, x::T) = setvalue!(θ, eta(θ.interval,x))

checktheta{T}(θ::HyperParameter, x::T) = checktheta(θ.interval, x)

getderiv_eta(θ::HyperParameter) = deriv_eta(θ.interval, getvalue(θ))

for op in (:isless, :(==), :+, :-, :*, :/)
    @eval begin
        $op(θ1::HyperParameter, θ2::HyperParameter) = $op(getvalue(θ1), getvalue(θ2))
        $op(a::Number, θ::HyperParameter) = $op(a, getvalue(θ))
        $op(θ::HyperParameter, a::Number) = $op(getvalue(θ), a)
    end
end
###Old version not using reparametrization
update!{T}(param::HyperParameter{T},grad::T) = begin
    # println("Correc : $(getderiv_eta(param)), Grad : $(GradDescent.update(param.opt,grad)), theta : $(gettheta(param))")
    isfree(param) ? settheta!(param, gettheta(param) + update(param.opt,getderiv_eta(param)*grad)) : nothing
    # isfree(param) ? setvalue!(param, getvalue(param) + GradDescent.update(param.opt,grad)) : nothing
end

isfree(θ::HyperParameter) = !θ.fixed

setfixed!(θ::HyperParameter) = θ.fixed = true



setfree!(θ::HyperParameter) = θ.fixed = false

mutable struct HyperParameters{T<:AbstractFloat}
    hyperparameters::Array{HyperParameter{T},1}
    function HyperParameters{T}(θ::Vector{T},intervals::Array{Interval{T,A,B}}) where {A<:Bound{T},B<:Bound{T}} where {T<:AbstractFloat}
        this = new(Vector{HyperParameter{T}}())
        for (val,int) in zip(θ,intervals)
            push!(this.hyperparameters,HyperParameter{T}(val,int))
        end
        return this
    end
end
function HyperParameters(θ::Vector{T},intervals::Vector{Interval{T,A,B}}) where {A<:Bound{T},B<:Bound{T}} where {T<:Real}
    HyperParameters{T}(θ,intervals)
end

@inline getvalue(θ::HyperParameters) = broadcast(getvalue,θ.hyperparameters)

function Base.getindex(p::HyperParameters,it::Integer)
    return p.hyperparameters[it]
end

function update!(param::HyperParameters,grad)
    for i in 1:length(param.hyperparameters)
        update!(param.hyperparameters[i],grad[i])
    end
end

setfixed!(θ::HyperParameters) = setfixed!.(θ.hyperparameters)

setfree!(θ::HyperParameters) = setfree!.(θ.hyperparameters)
