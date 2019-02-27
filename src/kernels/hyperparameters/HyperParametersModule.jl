module HyperParametersModule
import Base: convert, eltype, promote_type, show, string, ==, *, /, +, -, ^, isless;
using GradDescent
export
    Bound,
        LeftBound,
        RightBound,
        NullBound,

    Interval,
    interval,
    checkbounds,

    HyperParameter,
    HyperParameters,
    update!,
    getvalue,
    setvalue!,
    checkvalue,
    gettheta,
    settheta!,
    checktheta,
    upperboundtheta,
    lowerboundtheta,
    setfixed!,
    setfree!,
    setparamoptimizer!

include("bound.jl")
include("interval.jl")
include("hyperparameter.jl")

end # End HyperParameter
