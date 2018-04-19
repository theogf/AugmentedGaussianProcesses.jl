
using Distributions
include("../src/OMGP.jl")
import OMGP

N_data = 200
N_test = 20
N_dim = 2
noise = 0.2
minx=-5.0
maxx=5.0
function latent(x)
    return x[:,1].*sin.(x[:,2])
end
X = rand(N_data,N_dim)*(maxx-minx)+minx
x_test = linspace(minx,maxx,N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
y = latent(X)+rand(Normal(0,noise),size(X,1))
y_test = latent(X_test)
(nSamples,nFeatures) = (N_data,1)
kernel = OMGP.RBFKernel(2.0)
t_full = @elapsed fullmodel = OMGP.GPRegression(X,y,noise=noise,kernel=kernel,VerboseLevel=3)
t_sparse = @elapsed sparsemodel = OMGP.SparseGPRegression(X,y,Stochastic=false,Autotuning=false,VerboseLevel=3,m=20,noise=noise,kernel=kernel)
t_stoch = @elapsed stochmodel = OMGP.SparseGPRegression(X,y,Stochastic=true,BatchSize=20,Autotuning=true,VerboseLevel=2,m=20,noise=noise,kernel=kernel)
t_full += @elapsed fullmodel.train()
t_sparse += @elapsed sparsemodel.train(iterations=20)
t_stoch += @elapsed stochmodel.train(iterations=200)
y_full = fullmodel.predict(X_test); rmse_full = norm(y_full-y_test,2)/sqrt(length(y_test))
y_sparse = sparsemodel.predict(X_test); rmse_sparse = norm(y_sparse-y_test,2)/sqrt(length(y_test))
y_stoch = stochmodel.predict(X_test); rmse_stoch = norm(y_stoch-y_test,2)/sqrt(length(y_test))

println("Full model : RMSE=$(rmse_full), time=$t_full")
println("Sparse model : RMSE=$(rmse_sparse), time=$t_sparse")
println("Stoch. Sparse model : RMSE=$(rmse_stoch), time=$t_stoch")
using Plots
p1=plot(x_test,x_test,reshape(latent(X_test),N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="")
plot!(p1,X[:,1],X[:,2],t=:scatter,lab="training points",title="Truth")
p2=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="Regression")
p3=plot(x_test,x_test,reshape(y_sparse,N_test,N_test),t=:contour,fill=true,cbar=false,clims=[-5,5],lab="",title="Sparse Regression")
plot!(sparsemodel.inducingPoints[:,1],sparsemodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
p4=plot(x_test,x_test,reshape(y_stoch,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(minx*1.1,maxx*1.1),lab="",title="Stoch. Sparse Regression")
plot!(stochmodel.inducingPoints[:,1],stochmodel.inducingPoints[:,2],t=:scatter,lab="inducing points")
display(plot(p1,p2,p3,p4))
