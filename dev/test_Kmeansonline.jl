using Distributions
using Plots
using Clustering
using LinearAlgebra
pyplot()
using AugmentedGaussianProcesses
const AGP = AugmentedGaussianProcesses
println("Packages loaded")

function generate_random_walk_data(N,dim,noise)
    X = zeros(N,dim)
    f = zeros(N)
    df = Normal(0,0.01)
    if dim == 1
        d = Normal(0,noise)
    else
        d = MvNormal(zeros(dim),noise)
    end
    for i in 2:N
        X[i,:] = X[i-1,:].+rand(d)
        f[i] = f[i-1]+rand(df)
    end
    return X,f
end

l = 0.5
noise = 0.1
function distance(X,C)
    # return norm(X-C,2)^2
    return 2*(1-exp(-0.5*norm(X-C)^2/(l^2)))
end

kernel = RBFKernel(l)

function sample_gaussian_process(X,noise)
    N = size(X,1)
    K = kernelmatrix(X,kernel)+noise*Diagonal{Float64}(I,N)
    return rand(MvNormal(zeros(N),K))
end
dim = 2
n = 300
X,f = generate_random_walk_data(n,dim,0.1)
# X = readdlm("data/banana_X_train"); n,d=size(X)
# plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
f= sample_gaussian_process(X,0.1)

k = 64
b = 10
dorand = false
if dorand
    println("Randomizing the $n points first")
    randord = shuffle(1:n)
    X = X[randord,:]
    # f = f[randord]
end

N_test = 50
x1_grid = range(minimum(X[:,1])*1.1,stop=maximum(X[:,1])*1.1,length=N_test)
x2_grid = range(minimum(X[:,2])*1.1,stop=maximum(X[:,2])*1.1,length=N_test)
X_test = hcat([i for i in x1_grid, j in x2_grid][:],[j for i in x1_grid, j in x2_grid][:])

##### Circles threshold
lim = 0.8
circlealg = CircleKMeans(lim)
init!(circlealg,X,f,kernel)
for i in 1:n
    update!(circlealg,reshape(X[i,:],1,size(X,2)),f[i],kernel)
end
##### Custom implementation
Plots.plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p4 = Plots.plot!(circlealg.centers[:,1],circlealg.centers[:,2],t=:scatter,lab="",marker=:o,title="Custom K ($(circlealg.k)) ")
println("Custom_k cost $(total_cost(X,circlealg.centers,kernel)) with $(circlealg.k) clusters")

p5 = Plots.plot(-5:0.01:5,x->1-0.5*distance(x,0),lab="k(x,0)",linewidth=2.0)
Plots.plot!(x->lim,lab="Threshold",linewidth=3.0)


#### Using DeterminantalPointProcesses
dppalg = DPPAlg(0.05,kernel)
init!(dppalg,X,f,kernel)
update!(dppalg,X,f,kernel)
Plots.plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p6 = Plots.plot!(dppalg.centers[:,1],dppalg.centers[:,2],t=:scatter,lab="",marker=:o,title="DPP ($(dppalg.k)) ")
println("DPP cost $(total_cost(X,dppalg.centers,kernel)) with $(dppalg.k) clusters")

### DPP + Webscale
c = copy(dppalg.centers)
dppweb = Webscale(dppalg.k)
init!(dppweb,X,f,kernel)
dppweb.centers=c
for _ in 1:10
    update!(dppweb,X,f,kernel)
end


Plots.plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p7 = Plots.plot!(dppweb.centers[:,1],dppweb.centers[:,2],t=:scatter,lab="",marker=:o,title="DPP+WEB ($(dppweb.k)) ")
println("DPP+WEB cost $(total_cost(X,dppweb.centers,kernel)) with $(dppweb.k) clusters")

k = dppweb.k
###Basic Offline KMeans
offkmeans = OfflineKmeans(k)
init!(offkmeans,X,nothing,kernel)
update!(offkmeans,X,nothing,nothing)
Plots.plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p1 = Plots.plot!(offkmeans.centers[:,1],offkmeans.centers[:,2],t=:scatter,color=:red,lab="",title="Offline Kmeans")
println("Full kmeans cost : $(total_cost(X,offkmeans.centers,kernel)) with $(offkmeans.k) clusters")
###### Web-scale k-means clustering #####

web = Webscale(k)
init!(web,X,f,kernel)
T=10000
onepass=false
d = zeros(Int64,b)
if onepass
    iter = 1
    while (iter+b) < n
        update!(web,X[iter:(iter+b),:],nothing,kernel)
        global iter += b
    end
else
    iter = 1
    while iter < T
        samples = sample(1:n,b,replace=false)
        update!(web,X[samples,:],f,kernel )
        global iter+=1
    end
end
Plots.plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p2 = Plots.plot!(web.centers[:,1],web.centers[:,2],t=:scatter,lab="",marker=:o,title="Webscale ($(web.k))")
println("Webscale cost : $(total_cost(X,web.centers,kernel)) with $(web.k) clusters")
###### An algorithm for Online K Means clustering
##### The pragmatic online algorithm
stream = StreamOnline(k)
init!(stream,X,f,kernel)

for i in (k+1):2:n
    update!(stream,X[i:i+1,:],f,nothing)
end
Plots.plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p3 = Plots.plot!(stream.centers[:,1],stream.centers[:,2],t=:scatter,lab="",marker=:o,title="Streaming Online ($(stream.k))")
println("Streaming Online cost $(total_cost(X,stream.centers,kernel)) with $(stream.k) clusters")



display(Plots.plot(p1,p2,p3,p4,p5,p6,p7))





function get_sigma(k_starstar,centers,X_test,kernel,noise)
# centers = offkmeans.centers
    k_star = kernelmatrix(X_test,centers,kernel)
    k_starstar = kerneldiagmatrix(X_test,kernel)
    K = kernelmatrix(centers,kernel)
    return k_starstar-sum((k_star*inv(K+noise*I)).*k_star,dims=2)
end

# p1 = plot(x1_grid,x2_grid,reshape(σ_off,N_test,N_test)',t=:contourf)
σ_true = get_sigma(k_starstar,X,X,kernel,noise)
p1 = plot(x1_grid,x2_grid,reshape(get_sigma(k_starstar,X,X_test,kernel,noise),N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
# plot!(offkmeans.centers[:,1],offkmeans.centers[:,2],t=:scatter,lab="",marker=:o,title="Offline KMeans ($(offkmeans.k)) ")

σ_off = get_sigma(k_starstar,offkmeans.centers,X_test,kernel,noise)
p2 = plot(x1_grid,x2_grid,reshape(σ_off,N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
plot!(offkmeans.centers[:,1],offkmeans.centers[:,2],t=:scatter,lab="",marker=:o,title="Offline KMeans ($(web.k)) ")
println("Offline kmeans median error on σ : $(median(get_sigma(k_starstar,offkmeans.centers,X,kernel,noise)-σ_true))")

σ_web = get_sigma(k_starstar,web.centers,X_test,kernel,noise)
p3 = plot(x1_grid,x2_grid,reshape(σ_web,N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
plot!(web.centers[:,1],web.centers[:,2],t=:scatter,lab="",marker=:o,title="Webscale KMeans ($(web.k)) ")
println("Webscale median error on σ : $(median(get_sigma(k_starstar,web.centers,X,kernel,noise)-σ_true))")

σ_str = get_sigma(k_starstar,stream.centers,X_test,kernel,noise)
p4 = plot(x1_grid,x2_grid,reshape(σ_str,N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
plot!(stream.centers[:,1],stream.centers[:,2],t=:scatter,lab="",marker=:o,title="Streaming KMeans ($(stream.k)) ")
println("Streaming median error on σ : $(median(get_sigma(k_starstar,stream.centers,X,kernel,noise)-σ_true))")

σ_cust = get_sigma(k_starstar,circlealg.centers,X_test,kernel,noise)
p5 = plot(x1_grid,x2_grid,reshape(σ_cust,N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
plot!(circlealg.centers[:,1],circlealg.centers[:,2],t=:scatter,lab="",marker=:o,title="Circle KMeans ($(circlealg.k)) ")
println("Circle kmeans median error on σ : $(median(get_sigma(k_starstar,stream.centers,X,kernel,noise)-σ_true))")

σ_dpp = get_sigma(k_starstar,dppalg.centers,X_test,kernel,noise)
p6 = plot(x1_grid,x2_grid,reshape(σ_dpp,N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
plot!(dppalg.centers[:,1],dppalg.centers[:,2],t=:scatter,lab="",marker=:o,title="DPP ($(dppalg.k)) ")
println("DPP median error on σ : $(median(get_sigma(k_starstar,dppalg.centers,X,kernel,noise)-σ_true))")


σ_dppweb = get_sigma(k_starstar,dppweb.centers,X_test,kernel,noise)
p7 = plot(x1_grid,x2_grid,reshape(σ_dppweb,N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
plot!(dppweb.centers[:,1],dppweb.centers[:,2],t=:scatter,lab="",marker=:o,title="DPP+WEB ($(dppweb.k)) ")
println("DPP+WEB median error on σ : $(median(get_sigma(k_starstar,dppweb.centers,X,kernel,noise)-σ_true))")



display(plot(p1,p2,p3,p4,p5,p6,p7))
# end
