import OMGP
using Distributions
using StatsBase
using Gallium
N_data = 500
N_class = 4
N_test = 100
minx=-5.0
maxx=5.0
noise = 0.5


function latent(X)
    return sqrt.(X[:,1].^2+X[:,2].^2)
end

# function get_y(X)


# X = (rand(N_data,2)*(maxx-minx))+minx
X = rand(Normal(0,3),N_data,2)
x_test = linspace(minx,maxx,N_test)
X_test = hcat([j for i in x_test, j in x_test][:],[i for i in x_test, j in x_test][:])
y = min.(max.(1,floor.(Int64,latent(X)+rand(Normal(0,noise),size(X,1)))),N_class)
y_test =  min.(max.(1,floor.(Int64,latent(X_test))),N_class)

# data = readdlm("data/Iris")
#Dataset has been already randomized
# X = data[1:100,1:(end-1)]; y=data[1:100,end]
# X_test = data[101:end,1:(end-1)]; y_test=data[101:end,end]

kernel = OMGP.RBFKernel(1.0)
# kernel= OMGP.PolynomialKernel([1.0,0.0,1.0])
model = OMGP.MultiClass(X,y,VerboseLevel=1,kernel=kernel)
t_full = @elapsed model.train(iterations=200)
y_full = model.predict(X_test)
println(t_full)
# conf_matrix = zeros(N_class,N_class)
# for i in 1:(N_test^2)
#     conf_matrix[y_full[i],y_test[i]] += 1
# end
# println("Accuracy is $(trace(conf_matrix)/sum(conf_matrix))")

score = 0
for (i,pred) in enumerate(y_full)
    if pred == y_test[i]
        score += 1
    end
end
println("Accuracy is $(score/length(y_test))")


using Plots
plotlyjs()
p1=plot(x_test,x_test,reshape(y_test,N_test,N_test),t=:contour,clims=(1,N_class),cbar=false,fill=:true)
[plot!(X[y.==i,1],X[y.==i,2],t=:scatter,lab="y=$i",title="Truth",xlims=(-5,5),ylims=(-5,5)) for i in 1:N_class]
p2=plot(x_test,x_test,reshape(y_full,N_test,N_test),t=:contour,clims=(1,N_class),fill=true,cbar=false,lab="",title="MultiClass")
display(plot(p1,p2));
