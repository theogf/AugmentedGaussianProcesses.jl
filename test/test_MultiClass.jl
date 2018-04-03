import OMGP
using Distributions
using StatsBase
using Gallium
N_data = 500
N_class = 4
N_test = 100
minx=-5.0
maxx=5.0
noise = 0.1


function latent(X)
    return sqrt.(X[:,1].^2+X[:,2].^2)
end

# function get_y(X)


# X = (rand(N_data,2)*(maxx-minx))+minx
# trunc_d = Truncated(Normal(0,3),minx,maxx)
# X = rand(trunc_d,N_data,2)
# x_test = linspace(minx,maxx,N_test)
# X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
# y = min.(max.(1,floor.(Int64,latent(X)+rand(Normal(0,noise),size(X,1)))),N_class)
# y_test =  min.(max.(1,floor.(Int64,latent(X_test))),N_class)

# data = readdlm("data/Iris")
#Dataset has been already randomized
# X = data[1:100,1:(end-1)]; y=data[1:100,end]
# X_test = data[101:end,1:(end-1)]; y_test=data[101:end,end]


data = readdlm("data/Glass")
# Dataset has been already randomized
X = data[1:150,1:(end-1)]; y=data[1:150,end]
X_test = data[151:end,1:(end-1)]; y_test=data[151:end,end]

# X = readdlm("data/mnist_train")
# y=  X[:,1]; X= X[:,2:end]
# X_test = readdlm("data/mnist_test")
# y_test= X_test[:,1]; X_test=X_test[:,2:end]

kernel = OMGP.RBFKernel(0.1)
# kernel= OMGP.PolynomialKernel([1.0,0.0,1.0])
full_model = OMGP.MultiClass(X,y,VerboseLevel=1,kernel=kernel)
sparse_model = OMGP.SparseMultiClass(X,y,VerboseLevel=3,kernel=kernel,m=100,Stochastic=false)
t_full = @elapsed full_model.train(iterations=200)
t_sparse = @elapsed sparse_model.train(iterations=200)
y_full = full_model.predict(X_test)
y_sparse = sparse_model.predict(X_test)
# conf_matrix = zeros(N_class,N_class)
# for i in 1:(N_test^2)
#     conf_matrix[y_full[i],y_test[i]] += 1
# end
# println("Accuracy is $(trace(conf_matrix)/sum(conf_matrix))")

full_score = 0
for (i,pred) in enumerate(y_full)
    if pred == y_test[i]
        full_score += 1
    end
end
sparse_score=0
for (i,pred) in enumerate(y_sparse)
    if pred == y_test[i]
        sparse_score += 1
    end
end
println("Full model Accuracy is $(full_score/length(y_test)) in $t_full s")
println("Sparse model Accuracy is $(sparse_score/length(y_test)) in $t_sparse s")

#
# using Plots
# # plotlyjs()
# p1=plot(x_test,x_test,reshape(y_test,N_test,N_test),t=:contour,clims=(1,N_class),cbar=false,fill=:true)
# [plot!(X[y.==i,1],X[y.==i,2],t=:scatter,lab="y=$i",title="Truth",xlims=(-5,5),ylims=(-5,5)) for i in 1:N_class]
# p2=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="MultiClass")
# p3=plot(x_test,x_test,reshape(y_sparse,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="Sparse MultiClass ($(sparse_model.m) points)")
# [plot!(model.inducingPoints[k][:,1],model.inducingPoints[k][:,2],t=:scatter,lab="y=$(model.class_mapping[k])",xlims=(-5,5),y_lims=(-5,5)) for k in 1:N_class]
# display(plot(p1,p2,p3));
#
#
# full_f_star = OMGP.fstar(full_model,X_test,covf=false)
# sparse_f_star = OMGP.fstar(sparse_model,X_test,covf=false)
# p_full = [plot(x_test,x_test,reshape(full_f_star[i],N_test,N_test),t=:contour,fill=true,cbar=false,lab="",title="f_$(model.class_mapping[i])") for i in 1:N_class]
# p_sparse = [begin plot(x_test,x_test,reshape(sparse_f_star[i],N_test,N_test),t=:contour,fill=true,cbar=false,lab="",title="f_$(model.class_mapping[i])");plot!(model.inducingPoints[i][:,1],model.inducingPoints[i][:,2],t=:scatter,lab="") end for i in 1:N_class]
# plot(p_full...,p_sparse...,layout=[N_class,2])
