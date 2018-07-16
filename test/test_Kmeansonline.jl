using Distributions
using Plots
using Clustering
pyplot()
include("../src/OMGP.jl")
using OMGP.KernelFunctions
using OMGP.KMeansModule
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
        X[i,:] = X[i-1,:]+rand(d)
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

kernel = KernelFunctions.RBFKernel(l)

function sample_gaussian_process(X,noise)
    N = size(X,1)
    K = KernelFunctions.kernelmatrix(X,kernel)+noise*eye(N)
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
x1_grid = linspace(minimum(X[:,1]),maximum(X[:,1]),N_test)
x2_grid = linspace(minimum(X[:,2]),maximum(X[:,2]),N_test)
X_test = hcat([i for i in x1_grid, j in x2_grid][:],[j for i in x1_grid, j in x2_grid][:])

###Basic Offline KMeans
offkmeans = KMeansModule.OfflineKmeans()
KMeansModule.init!(offkmeans,X,k)
KMeansModule.update!(offkmeans,X)
plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p1 = plot!(offkmeans.centers[:,1],offkmeans.centers[:,2],t=:scatter,color=:red,lab="",title="Offline Kmeans")
println("Full kmeans cost : $(KMeansModule.total_cost(X,offkmeans.centers)) with $(offkmeans.k) clusters")
###### Web-scale k-means clustering #####

web = KMeansModule.Webscale()
KMeansModule.init!(web,X,k)
i = 1
T=10000
onepass=true
d = zeros(Int64,b)
if onepass
    while (i+b) < n
        KMeansModule.update!(web,X[i:(i+b),:])
        i += b
    end
else
    while i < T
        samples = sample(1:n,b,replace=false)
        KMeansModule.update!(web,X[samples,:])
        i+=1
    end
end
plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p2 = plot!(web.centers[:,1],web.centers[:,2],t=:scatter,lab="",marker=:o,title="Webscale ($(web.k))")
println("Webscale cost : $(KMeansModule.total_cost(X,web.centers)) with $(web.k) clusters")
###### An algorithm for Online K Means clustering
##### The pragmatic online algorithm
stream = KMeansModule.StreamOnline()
KMeansModule.init!(stream,X,k)

for i in (k+1):2:n
    KMeansModule.update!(stream,X[i:i+1,:])
end
plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p3 = plot!(stream.centers[:,1],stream.centers[:,2],t=:scatter,lab="",marker=:o,title="Streaming Online ($(stream.k))")
println("Streaming Online cost $(KMeansModule.total_cost(X,stream.centers)) with $(stream.k) clusters")


function find_nearest_center(X,C)
    nC = size(C,1)
    best = Int64(1); best_val = Inf
    for i in 1:nC
        val = distance(X,C[i,:])
        if val < best_val
            best_val = val
            best = i
        end
    end
    return best,best_val
end



function total_cost(X,C)
    n = size(X,1)
    tot = 0
    for i in 1:n
        tot += find_nearest_center(X[i,:],C)[2]
    end
    return tot
end

lim = 0.5

C = reshape(X[1,:],1,2)
k = size(C,1)
K_C = KernelFunctions.kernelmatrix(C,kernel)
ind_C = [1]
s_diff = zeros(size(X,1))
s_d = zeros(size(X,1))
s_diff[1]=0.0
s_d[1] = 0.0
count = 0
for i in 2:size(X,1)
    # j,d = find_nearest_center(X[i,:],C)
    # if 0.5*sqrt(d)>rand()
    # diff = norm(f[i]-f[j],2)/abs(0.5*(f[i]+f[j]))
    # s_diff[i]=diff
    # s_d[i]=1-0.5*d
    K_star = KernelFunctions.kernelmatrix(reshape(X[i,:],1,2),C,kernel)
    k_starstar = KernelFunctions.diagkernelmatrix(reshape(X[i,:],1,2),kernel)
    invK = inv(K_C+noise*eye(k))
    σ = (k_starstar[1]-((K_star*invK)*transpose(K_star))[1])/(k_starstar[1])
    μ = K_star*invK*f[ind_C]
    diff_f = 1-exp(-0.5*(μ[1]-f[i])^2/σ)
    # lim = (tanh(diff)+1.0)*0.5
    # println("k : $(1-0.5*d) vs lim : $lim ")
    # σ_cust = get_sigma(KernelFunctions.diagkernelmatrix(X_test,kernel),C,X_test,kernel,noise)
    # plot(x1_grid,x2_grid,reshape(σ_cust,N_test,N_test)',t=:contourf)
    # plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
    # plot!([X[i,1]],[X[i,2]],t=:scatter,lab="",alpha=1.0,color=:red,markerstrokewidth=0)
    # display(plot!(C[:,1],C[:,2],t=:scatter,lab="",marker=:o,title="Custom K ($(k))\n σ = $(σ) "))
    if diff_f > sqrt(σ)
        # println("$count : $diff_f, $σ")
        count += 1
    end
    if (0.8*σ+0.2*diff_f)>rand()
    # if σ>diff_f
    # if σ
    # if d>2*(1-lim)
        C = vcat(C,X[i,:]')
        push!(ind_C,i)
        K_C = KernelFunctions.kernelmatrix(C,kernel)
        k = size(C,1)
    end
end
cust_k = size(C,1)
##### Custom implementation
plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p4 = plot!(C[:,1],C[:,2],t=:scatter,lab="",marker=:o,title="Custom K ($(cust_k)) ")
println("Custom_k cost $(total_cost(X,C)) with $(cust_k) clusters")

### Full plotting
p5 = plot(-5:0.01:5,x->1-0.5*distance(x,0),lab="k(x,0)",linewidth=2.0)
plot!(x->lim,lab="Threshold",linewidth=3.0)
p6  =  plot(X[:,1],X[:,2],f[:],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0)
plot!(C[:,1],C[:,2],f[ind_C],t=:scatter,lab="",marker=:o,title="Custom K ($(cust_k)) ")
display(plot(p1,p2,p3,p4,p5,p6))

#
# if true


k_starstar = KernelFunctions.diagkernelmatrix(X_test,kernel)

function get_sigma(k_starstar,centers,X_test,kernel,noise)
# centers = offkmeans.centers
    k_star = KernelFunctions.kernelmatrix(X_test,centers,kernel)
    K = KernelFunctions.kernelmatrix(centers,kernel)
    return k_starstar-sum((k_star*inv(K+noise*eye(K))).*k_star,2)
end

σ_off = get_sigma(k_starstar,offkmeans.centers,X_test,kernel,noise)
# p1 = plot(x1_grid,x2_grid,reshape(σ_off,N_test,N_test)',t=:contourf)
σ_true = get_sigma(k_starstar,X,X_test,kernel,noise)
p1 = plot(x1_grid,x2_grid,reshape(σ_true,N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
# plot!(offkmeans.centers[:,1],offkmeans.centers[:,2],t=:scatter,lab="",marker=:o,title="Offline KMeans ($(offkmeans.k)) ")
println("Off kmeans median error on σ : $(median(σ_off-σ_true))")

σ_web = get_sigma(k_starstar,web.centers,X_test,kernel,noise)
p2 = plot(x1_grid,x2_grid,reshape(σ_web,N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
plot!(web.centers[:,1],web.centers[:,2],t=:scatter,lab="",marker=:o,title="Webscale KMeans ($(web.k)) ")
println("Webscale median error on σ : $(median(σ_web-σ_true))")

σ_str = get_sigma(k_starstar,stream.centers,X_test,kernel,noise)
p3 = plot(x1_grid,x2_grid,reshape(σ_str,N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
plot!(offkmeans.centers[:,1],offkmeans.centers[:,2],t=:scatter,lab="",marker=:o,title="Streaming KMeans ($(stream.k)) ")
println("Streaming median error on σ : $(median(σ_str-σ_true))")

σ_cust = get_sigma(k_starstar,C,X_test,kernel,noise)
p4 = plot(x1_grid,x2_grid,reshape(σ_cust,N_test,N_test)',t=:contourf)
plot!(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.5,markerstrokewidth=0,color=:blue)
plot!(C[:,1],C[:,2],t=:scatter,lab="",marker=:o,title="Custom KMeans ($(k)) ")
println("Custom kmeans median error on σ : $(median(σ_cust-σ_true))")

display(plot(p1,p2,p3,p4))
# end
