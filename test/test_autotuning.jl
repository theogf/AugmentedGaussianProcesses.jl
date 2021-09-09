using Test
using FiniteDifferences
using Random: seed!
using Zygote, Flux, ForwardDiff, LinearAlgebra
using AugmentedGaussianProcesses;
const AGP = AugmentedGaussianProcesses;
@info "Packages loaded"
seed!(42)
M = 10
N = 100
D = 2
T = 3000
fullgp = false
X = rand(N, D) * 2 .- 1.0
y = norm.(eachrow(X)) .- 0.5#randn(N)
xgrid = range(-1, 1; length=100)
Xtest = vcat(transpose.(collect.(Iterators.product(xgrid, xgrid)))...)
kernel = SqExponentialKernel(3.0)
# kernel = SqExponentialKernel([3.0,4.0])
Z_init = rand(M, D) * 2 .- 1.0

##
AGP.setadbackend(:reverse_diff)
grads_reverse = []
ELBO_reverse = []
l_reverse = []
σ_reverse = []
function cb_reverse(model, iter)
    if iter > 3
        push!(grads_reverse, AGP.grads[first(Flux.params(model.f[1].kernel))])
    end
    push!(ELBO_reverse, ELBO(model))
    return push!(l_reverse, first(model.f[1].kernel.transform.s))
end
model = if fullgp
    VGP(X, sign.(y), deepcopy(kernel), LogisticLikelihood(), AnalyticVI())
else
    SVGP(X, sign.(y), deepcopy(kernel), LogisticLikelihood(), AnalyticVI(), M)
end

if !fullgp
    model.f[1].Z.Z .= copy(Z_init)
end
@time train!(model, T; callback=cb_reverse)

##
AGP.setadbackend(:forward_diff)
grads_forward = []
gradσ_forward = []
ELBO_forward = []
l_forward = []
σ_forward = []
function cb_forward(model, iter)
    if iter > 3
        push!(grads_forward, AGP.grads[first(Flux.params(model.f[1].kernel))])
    end
    push!(ELBO_forward, ELBO(model))
    return push!(l_forward, first(model.f[1].kernel.transform.s))
end
# model = OnlineSVGP(deepcopy(kernel),LogisticLikelihood(),AnalyticVI(),)
model = if fullgp
    VGP(X, sign.(y), deepcopy(kernel), LogisticLikelihood(), AnalyticVI())
else
    SVGP(X, sign.(y), deepcopy(kernel), LogisticLikelihood(), AnalyticVI(), M)
end

if !fullgp
    model.f[1].Z.Z .= copy(Z_init)
end
@time train!(model, T; callback=cb_forward)
# @time train!(model,X,sign.(y),iterations=20)

##
using Plots;
pyplot()
default(; lw=3.0)
y_pred, _ = proba_y(model, Xtest)

p = plot(; layout=(3, 2))
for j in 1:length(grads_forward[1])
    plot!(1:length(grads_forward), getindex.(grads_forward, j); lab="forward", subplot=1)
    plot!(
        1:length(grads_reverse),
        getindex.(grads_reverse, j);
        title="D=$j",
        linestyle=:dash,
        lab="reverse",
        subplot=1,
    )
    plot!(1:length(ELBO_forward), ELBO_forward; lab="forward", subplot=2)
    plot!(
        1:length(ELBO_reverse);
        ls=:dash,
        ELBO_reverse,
        lab="reverse",
        title="ELBO",
        subplot=2,
    )
    plot!(1:length(l_forward), l_forward; lab="forward", subplot=3)
    plot!(
        1:length(l_reverse);
        ls=:dash,
        l_reverse,
        lab="reverse",
        title="l",
        subplot=3,
        yaxis=:log,
    )
    Plots.contourf!(
        collect(xgrid),
        collect(xgrid),
        reshape(y_pred, 100, 100);
        subplot=4,
        clims=(0.0, 1.0),
    )
    Plots.scatter!(eachcol(X)...; zcolor=sign.(y), subplot=4, lab="")
    if !fullgp
        scatter!(eachcol(Z_init)...; color=:blue, subplot=4, lab="")
    end
    plot!(1:length(σ_forward), σ_forward; lab="forward", subplot=5)
    plot!(
        1:length(σ_reverse);
        ls=:dash,
        σ_reverse,
        lab="reverse",
        title="σ",
        subplot=5,
        yaxis=:log,
    )
    plot!(1:length(gradσ_forward), gradσ_forward; lab="forward", subplot=6)
    plot!(
        1:length(gradσ_reverse);
        ls=:dash,
        gradσ_reverse,
        lab="reverse",
        title="gradσ",
        subplot=6,
    )
end
display(p)

Zygote.refresh()
g = Zygote.gradient(() -> AGP.ELBO_given_theta(model), Flux.params(model.f[1].kernel))
AGP.update_hyperparameters!(model)
AGP.grads[first(Flux.params(model.f[1].kernel))][1]
g.grads[first(Flux.params(model.f[1].kernel))]
AGP.grads
g.grads

Flux.params(model.f[1].kernel)
