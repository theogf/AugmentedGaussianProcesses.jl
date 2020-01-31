using Test
using FiniteDifferences
using Zygote
using ForwardDiff
using AugmentedGaussianProcesses; const AGP = AugmentedGaussianProcesses

X = rand(10,2)
y = randn(10)
kernel = SqExponentialKernel([3.0,4.0])


##
AGP.setadbackend(:reverse_diff)
grads_reverse = []
cb_reverse(model,iter)= iter > 3 ? push!(grads_reverse,AGP.grads[first(Flux.params(model.f[1].kernel))]) : nothing
model =  VGP(X,sign.(y),deepcopy(kernel),LogisticLikelihood(),AnalyticVI())
@time train!(model,20,callback=cb_reverse)

##
AGP.setadbackend(:forward_diff)
grads_forward = []
cb_forward(model,iter)=iter > 3 ? push!(grads_forward,AGP.grads[first(Flux.params(model.f[1].kernel))]) : nothing
model = VGP(X,sign.(y),deepcopy(kernel),LogisticLikelihood(),AnalyticVI())
@time train!(model,20,callback=cb_forward)


##
using Plots
for j in 1:length(grads_forward[1])
    plot(1:length(grads_forward),getindex.(grads_forward,j),lab="forward")
    plot!(1:length(grads_reverse),getindex.(grads_reverse,j),title="D=$j",lab="reverse") |> display
end
