using Plots
using AugmentedGaussianProcesses
using LinearAlgebra, Distributions

N = 10; l = 0.1
N_grid = 500
Xgrid = collect(range(-0.2,1.2,length=N_grid))
X = rand(N,1)
mse(y,y_pred) = norm(y-y_pred)
ll(y,y_pred) =

K = kernelmatrix(X,RBFKernel(l))
μ_0 = 0.0*ones(N)# sin.(X[:]).+1.0
s = sortperm(X[:])
y_true= rand(MvNormal(μ_0,Symmetric(K+1e-1I)))
y = y_true + rand(TDist(3.0),N)
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

model = VStP(X,y,RBFKernel(0.1),GaussianLikelihood(0.01),AnalyticVI(),100.0,verbose=0,optimizer=true)
train!(model,iterations=500)#,callback=cplot)
p_y = predict_y(model,Xgrid)
# plot!(Xgrid,p_y,lab="")
# plot!(X[s],model.μ₀[1][s],lab="")
gpmodel = GP(X,y,RBFKernel(0.1),noise=0.01,verbose=0,optimizer=true)
train!(gpmodel,iterations=500)#,callback=cplot)
p_y = predict_y(model,Xgrid)
# plot!(Xgrid,p_y,lab="")
cplot(model,1)
cplot(gpmodel,1)

##
p_y,sig_y = proba_y(model,Xgrid)
p = scatter(X[s],y[s],lab="data")
p = plot!(Xgrid,p_y+2*sqrt.(sig_y),fill=p_y-2*sqrt.(sig_y),lab="",linewidth=0.0,alpha=0.2,color=1)
p = plot!(Xgrid,p_y,lab="Prediction T Process",color=1)

p_ygp,sig_ygp = proba_y(gpmodel,Xgrid)
p = plot!(Xgrid,p_ygp+2*sqrt.(sig_ygp),fill=p_ygp-2*sqrt.(sig_ygp),lab="",linewidth=0.0,alpha=0.2,color=2)
p = plot!(Xgrid,p_ygp,lab="Prediction G Process",color=2)

pk = scatter([getlengthscales(model.kernel[1])],[getvariance(model.kernel[1])],xlabel="Lengthscale",ylabel="Variance",lab="Student-T Process",xlims=(1e-3,1.0),ylims=(0.1,20.0),yaxis=:log,xaxis=:log)
scatter!(pk,[getlengthscales(gpmodel.kernel[1])],[getvariance(gpmodel.kernel[1])],lab="Gaussian Process",legend=:bottom)

plot(p,pk)
