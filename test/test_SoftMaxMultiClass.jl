using AugmentedGaussianProcesses
using Distributions
using StatsBase, Distances
using Random: seed!
using BenchmarkTools
using Dates
using PyCall
using ProfileView, Profile, Traceur
using ValueHistories
using Plots
pyplot()
seed!(42)
@pyimport sklearn.datasets as sk
@pyimport sklearn.model_selection as sp
N_data = 500
N_class = 3
N_test = 50
N_grid = 50
minx=-5.0
maxx=5.0
noise = 1.0
truthknown = false
doMCCompare = false
dolikelihood = false
println("$(now()): Starting testing multiclass")

function latent(X)
    return sqrt.(X[:,1].^2+X[:,2].^2)
end
N_dim=2
# X = (rand(N_data,N_dim)*(maxx-minx)).+minx
# trunc_d = Truncated(Normal(0,3),minx,maxx)
# X = rand(trunc_d,N_data,N_dim)
# x_test = range(minx,stop=maxx,length=N_test)
# X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
# # X_test = rand(trunc_d,N_test^dim,dim)
# y = min.(max.(1,floor.(Int64,latent(X)+rand(Normal(0,noise),size(X,1)))),N_class)
# y_test =  min.(max.(1,floor.(Int64,latent(X_test))),N_class)

# X,y = sk.make_classification(n_samples=N_data,n_features=N_dim,n_classes=N_class,n_clusters_per_class=1,n_informative=N_dim,n_redundant=0)
# y.+=1
# X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)

for c in 1:N_class
    global centers = rand(Uniform(-1,1),N_class,N_dim)
    global variance = 0.7*1/N_class*ones(N_class)#rand(Gamma(1.0,0.5),150)
end

X = zeros(N_data,N_dim)
y = sample(1:N_class,N_data)
for i in 1:N_data
    X[i,:] = rand(MvNormal(centers[y[i],:],variance[y[i]]))
end
xmin = minimum(X); xmax = maximum(X)
X,X_test,y,y_test = sp.train_test_split(X,y,test_size=0.33)
x_grid = range(xmin,length=N_grid,stop=xmax)
X_grid = hcat([j for i in x_grid, j in x_grid][:],[i for i in x_grid, j in x_grid][:])
# X,y = MNIST.traindata()
# X=Float64.(reshape(X,28*28,60000)')
# X_test,y_test = MNIST.testdata()
# X_test=Float64.(reshape(X_test,28*28,10000)')

#Test on the Iris dataet
# train = readdlm("data/iris-X")
# X = train[1:100,:]; X_test=train[101:end,:]
# test = readdlm("data/iris-y")
# y = test[1:100,:]; y_test=test[101:end,:]
# truthknown = false
#
# data = readdlm("data/Glass")
# # Dataset has been already randomized
# X = data[1:150,1:(end-1)]; y=data[1:150,end]
# X_test = data[151:end,1:(end-1)]; y_test=data[151:end,end]

# function norm_data(input_df)
#     for i in 1:size(input_df,2)
#         input_df[:,i] = (input_df[:,i] - mean(input_df[:,i])) ./
#             (var(input_df[:,i])==0?1:sqrt(var(input_df[:,i])))
#     end
#     return input_df
# end

#### Test on the mnist dataset
# X = readdlm("data/mnist_train")
# y=  X[:,1]; X= X[:,2:end]
# X_test = readdlm("data/mnist_test")
# y_test= X_test[:,1]; X_test=X_test[:,2:end]
# println("$(now()): MNIST data loaded")
#
# ### Test on the artificial character dataset
# X = readdlm("data/artificial-characters-train")

# y=  X[:,1]; X= X[:,2:end]
# X_test = readdlm("data/artificial-characters-test")
# y_test= X_test[:,1]; X_test=X_test[:,2:end]
# println("$(now()): Artificial Characters data loaded")



##Which algorithm are tested
fullm = !true
sfullm = !true
sparsem = true
ssparsem = !true
# for l in [0.001,0.005,0.01,0.05,0.1,0.5,1.0]
# for l in [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
function initial_lengthscale(X)
    D = pairwise(SqEuclidean(),X')
    return median([D[i,j] for i in 2:size(D,1) for j in 1:(i-1)])
end
 l = sqrt(initial_lengthscale(X))

kernel = AugmentedGaussianProcesses.RBFKernel([l],dim=N_dim)
# kernel = AugmentedGaussianProcesses.RBFKernel(l)


fmodel = AugmentedGaussianProcesses.SoftMaxMultiClass(X,y,verbose=3,noise=1e-3,ϵ=1e-20,kernel=kernel,Autotuning=false,AutotuningFrequency=2,IndependentGPs=true)
# fmetrics, callback = AugmentedGaussianProcesses.getMultiClassLog(fmodel,X_test=X_test,y_test=y_test)
# full_model.AutotuningFrequency=1
t_full = @elapsed fmodel.train(iterations=50,callback=callback)

global y_full,sig_full = fmodel.predict(X_test)
global y_fall = AugmentedGaussianProcesses.multiclasspredict(fmodel,X_test,true)
global y_ftrain, = fmodel.predict(X)
global y_fgrid, = fmodel.predict(X_grid)
println("Full predictions computed")
full_score = 0
for (i,pred) in enumerate(y_full)
    if pred == y_test[i]
        global full_score += 1
    end
end
println("Full model Accuracy is $(full_score/length(y_test)) in $t_full s for l = $l")
