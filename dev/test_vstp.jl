using Plots
using AugmentedGaussianProcesses
using LinearAlgebra, Distributions

N = 10; l = 0.1
N_grid = 500
Xgrid = collect(range(-0.2,1.2,length=N_grid))
X = rand(N,1)

K = kernelmatrix(X,RBFKernel(l))
μ_0 = 0.0*ones(N)# sin.(X[:]).+1.0
s = sortperm(X[:])
y= rand(MvNormal(μ_0,Symmetric(K+1e-1I)))
plot(X[s],y[s])
pk = plot()
function cplot(model,iter)
    global pk
    p_y,sig_y = proba_y(model,Xgrid)
    p = scatter(X[s],y[s],lab="data")
    plot!(Xgrid,p_y,lab="Prediction")
    p = plot!(Xgrid,p_y+2*sqrt.(sig_y),fill=p_y-2*sqrt.(sig_y),lab="",linewidth=0.0,alpha=0.2)
    pk = scatter!(pk,[getlengthscales(model.kernel[1])],[getvariance(model.kernel[1])],xlabel="Lengthscale",ylabel="Variance",lab="",xlims=(1e-3,1.0),ylims=(0.1,2.0),xaxis=:log)
    display(plot(p,pk))
end

model = VStP(X,y,RBFKernel(0.1),GaussianLikelihood(1.0),AnalyticVI(),3.0,verbose=0,optimizer=false)
train!(model,iterations=500)#,callback=cplot)
p_y = predict_y(model,Xgrid)
# plot!(Xgrid,p_y,lab="")
# plot!(X[s],model.μ₀[1][s],lab="")
gpmodel = GP(X,y,RBFKernel(0.1),noise=1.0,verbose=0,optimizer=false)
train!(gpmodel,iterations=100)#,callback=cplot)
p_y = predict_y(model,Xgrid)
# plot!(Xgrid,p_y,lab="")
cplot(model,1)
cplot(gpmodel,1)
