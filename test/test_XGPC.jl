using Distributions
using Gallium
include("../src/OMGP.jl")
import OMGP



### XOR dataset ###

N_data = 1000
N_test = 40
N_indpoints = 20
N_dim = 2
noise = 0.2
minx=-5.0
maxx=5.0

function latent(x)
    return x[:,1].*sin.(0.5*x[:,2])
end

X = rand(N_data,N_dim)*(maxx-minx)+minx
x_test = linspace(minx,maxx,N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
y = sign.(latent(X)+rand(Normal(0,noise),size(X,1)))
y_test = sign.(latent(X_test)+rand(Normal(0,noise),size(X_test,1)))
(nSamples,nFeatures) = (N_data,1)
kernel = OMGP.RBFKernel(1.0)

#### Banana dataset ####

# data = readdlm("banana_dataset")



# #### FULL MODEL EVALUATION ####
# t_full = @elapsed fullmodel = OMGP.BatchXGPC(X,y,noise=noise,kernel=kernel,VerboseLevel=3)
# t_full += @elapsed fullmodel.train()
# y_full = fullmodel.predictproba(X_test); acc_full = 1-sum(abs.(sign.(y_full-0.5)-y_test))/(2*length(y_test))
# p2=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(0,1),lab="",title="XGPC")
#
# #### SPARSE MODEL EVALUATION ####
# t_sparse = @elapsed sparsemodel = OMGP.SparseXGPC(X,y,Stochastic=false,Autotuning=true,VerboseLevel=3,m=N_indpoints,noise=noise,kernel=kernel)
# t_sparse += @elapsed sparsemodel.train(iterations=20)
# y_sparse = sparsemodel.predictproba(X_test); acc_sparse = 1-sum(abs.(sign.(y_sparse-0.5)-y_test))/(2*length(y_test))
# p3=plot(x_test,x_test,reshape(y_sparse,N_test,N_test),t=:contour,fill=true,cbar=false,clims=(0,1),lab="",title="Sparse XGPC")
# plot!(sparsemodel.inducingPoints[:,1],sparsemodel.inducingPoints[:,2],t=:scatter,lab="inducing points")

#### STOCH. SPARSE MODEL EVALUATION ###
t_stoch = @elapsed stochmodel = OMGP.SparseXGPC(X,y,Stochastic=true,BatchSize=40,Autotuning=true,VerboseLevel=2,m=N_indpoints,noise=noise,kernel=kernel)
metrics,savelog = OMGP.getLog(stochmodel)
OMGP.setfixed!(stochmodel.kernel.weight)
t_stoch += @elapsed stochmodel.train(iterations=10000,callback=savelog)
y_stoch = stochmodel.predictproba(X_test); acc_stoch = 1-sum(abs.(sign.(y_stoch-0.5)-y_test))/(2*length(y_test))
p4=plot(x_test,x_test,reshape(y_stoch,N_test,N_test),t=:contour,fill=true,cbar=true,clims=(0,1),lab="",title="Stoch. Sparse XGPC")
plot!(stochmodel.inducingPoints[:,1],stochmodel.inducingPoints[:,2],t=:scatter,lab="inducing points")

#### RESULTS OF THE ACCURACY ####
println("Full model : Acc=$(acc_full), time=$t_full")
println("Sparse model : Acc=$(acc_sparse), time=$t_sparse")
println("Stoch. Sparse model : Acc=$(acc_stoch), time=$t_stoch")

#### PRINTING RESULTS ####
using Plots
p1=plot(x_test,x_test,reshape(y_test,N_test,N_test),t=:contour,cbar=false,fill=:true)
plot!(X[y.==1,1],X[y.==1,2],color=:red,t=:scatter,lab="y=1",title="Truth",xlims=(-5,5),ylims=(-5,5))
plot!(X[y.==-1,1],X[y.==-1,2],color=:blue,t=:scatter,lab="y=-1")
display(plot(p1,p2,p3,p4));
