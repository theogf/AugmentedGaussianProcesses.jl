using Test
using FiniteDifferences
using Zygote
using ForwardDiff
using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses

M = 10
N = 100
D= 2
X = rand(N,D)
y = randn(N)
kernel = SqExponentialKernel([3.0,4.0])
Z_init = rand(M,D)

##
AGP.setadbackend(:reverse_diff)
grads_reverse = []
ELBO_reverse = []
function cb_reverse(model,iter)
    if iter > 3
        push!(grads_reverse,AGP.grads[first(Flux.params(model.f[1].kernel))])
    end
    push!(ELBO_reverse,ELBO(model))
end
model =  SVGP(X,sign.(y),deepcopy(kernel),LogisticLikelihood(),AnalyticVI(),10)
model.f[1].Z.Z .= copy(Z_init)
@time train!(model,20,callback=cb_reverse)

##
AGP.setadbackend(:forward_diff)
grads_forward = []
ELBO_forward = []
function cb_forward(model,iter)
    if iter > 3
        push!(grads_forward,AGP.grads[first(Flux.params(model.f[1].kernel))])
    end
    push!(ELBO_forward,ELBO(model))
end
model = OnlineSVGP(deepcopy(kernel),LogisticLikelihood(),AnalyticVI(),10)
# model = SVGP(X,sign.(y),deepcopy(kernel),LogisticLikelihood(),AnalyticVI(),10)
model.f[1].Z.Z .= copy(Z_init)
@time train!(model,20,callback=cb_forward)
@time train!(model,X,sign.(y),iterations=20)

##
using Plots
default(lw=3.0)
for j in 1:length(grads_forward[1])
    plot(1:length(grads_forward),getindex.(grads_forward,j),lab="forward")
    plot!(1:length(grads_reverse),getindex.(grads_reverse,j),title="D=$j",linestyle=:dash,lab="reverse") |> display
end
plot(1:length(ELBO_forward),ELBO_forward,lab="forward")
plot!(1:length(ELBO_reverse),ls=:dash,ELBO_reverse,lab="reverse",title="ELBO",)
