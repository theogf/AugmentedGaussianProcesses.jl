
using Distributions
include("../src/OMGP.jl")
import OMGP


N_data = 100
N_test = 20
N_dim = 2
noise = 0.2
function latent(x)
    return x[:,1].*sin.(x[:,2])
end
X_data = rand(N_data,N_dim)*10.0
x_test = linspace(0,10,N_test)
X_test = hcat(repmat(x_test, N_test, 1)[:], repmat(x_test, 1, N_test)[:])
y_data = sign.(latent(X_data)+rand(Normal(0,noise),size(X_data,1)))
y_test = latent(X_test)+rand(Normal(0,noise),size(X_test,1))
X=X_data; y=y_data
(nSamples,nFeatures) = (N_data,1)

model = OMGP.SparseXGPC(X,y,Stochastic=false,BatchSize=20,Autotuning=true,VerboseLevel=3,m=20,γ=noise,kernel=OMGP.RBFKernel(5.0))
model.train(iterations=1000)
y_pr = model.predict(X_test)
using Plots
plotlyjs()
p1=plot(x_test,x_test,latent(X_test),t=:contour,fill=true,cbar=true,lab="")
plot!(X_data[:,1],X_data[:,2],t=:scatter,lab="training")
plot(x_test,x_test,y_pr,t=:contour,fill=true,cbar=true,lab="")
p2=plot!(model.inducingPoints[:,1],model.inducingPoints[:,2],t=:scatter,lab="inducing points")
plot(p1,p2)
p = plot(x_test,x_test,abs.(latent(X_test)-y_pr),t=:contour,fill=true)
p = plot!(model.inducingPoints[:,1],model.inducingPoints[:,2],t=:scatter,lab="inducing points")
plot!(X_data[:,1],X_data[:,2],t=:scatter,lab="training")
