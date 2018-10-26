module HyperParametersModule
using AugmentedGaussianProcesses.GradDescent
import Base: convert, eltype, promote_type, show, string, ==, *, /, +, -, ^, isless;

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
    setfree!

include("bound.jl")
include("interval.jl")
include("hyperparameter.jl")

end # End HyperParameter
