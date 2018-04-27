module HyperParametersMod

import Base: convert, eltype, promote_type, show, string, ==, *, /, +, -, ^, isless

export
    Bound,
        LeftBound,
        RightBound,
        NullBound,

    Interval,
    interval,
    checkbounds,

    HyperParameter,
    getvalue,
    setvalue!,
    checkvalue,
    gettheta,
    settheta!,
    checktheta,
    upperboundtheta,
    lowerboundtheta

include("bound.jl")
include("interval.jl")
include("hyperparameter.jl")

end # End HyperParameter
