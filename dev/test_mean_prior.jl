using Plots
using AugmentedGaussianProcesses
using LinearAlgebra, Distributions

N = 300; l = 0.1
N_grid = 500
Xgrid = collect(range(-0.5,1.5,length=N_grid))
X = rand(N,1)

K = kernelmatrix(X,RBFKernel(l))
μ_0 = -100*ones(N)# sin.(X[:]).+1.0
s = sortperm(X[:])
y= rand(MvNormal(μ_0,Symmetric(K+1e-1I)))
plot(X[s],y[s])
pk = plot()
function cplot(model,iter)
    global pk
    p_y,sig_y = proba_y(model,Xgrid)
    p = scatter(X[s],y[s],lab="data")
    if typeof(model) <: SVGP
        ss = sortperm(model.Z[1][:])
        plot!(model.Z[1][ss],model.μ₀[1][ss],lab="Prior")
    else
        plot!(X[s],model.μ₀[1][s],lab="Prior")
    end
    plot!(Xgrid,p_y,lab="Prediction")
    p = plot!(Xgrid,p_y+2*sqrt.(sig_y),fill=p_y-2*sqrt.(sig_y),lab="",linewidth=0.0,alpha=0.2)
    pk = scatter!(pk,[getlengthscales(model.kernel[1])],[getvariance(model.kernel[1])],xlabel="Lengthscale",ylabel="Variance",lab="",xlims=(1e-3,1.0),ylims=(0.1,2.0),xaxis=:log)
    display(plot(p,pk))
end

model = SVGP(X,y,RBFKernel(0.1),GaussianLikelihood(0.1),AnalyticVI(),10,verbose=3)
train!(model,iterations=100,callback=cplot)
p_y = predict_y(model,Xgrid)
plot!(Xgrid,p_y,lab="")
# plot!(X[s],model.μ₀[1][s],lab="")
