using Distributions
using Plots
using Clustering
pyplot()
include("../src/OMGP.jl")
# import OMGP

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
dim = 2
n = 10000
# X,f = generate_random_walk_data(n,dim,0.1)
X = readdlm("data/banana_X_train"); n,d=size(X)
plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)

k = 256
b = 10
dorand = false
if dorand
    println("Randomizing the $n points first")
    randord = shuffle(1:n)
    X = X[randord,:]
    # f = f[randord]
end
###Basic Offline KMeans
offkmeans = OfflineKmeans()
init!(offkmeans,X,k)
update!(offkmeans,X)
plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p1 = plot!(offkmeans.centers[:,1],offkmeans.centers[:,2],t=:scatter,color=:red,lab="",title="Offline Kmeans")
println("Full kmeans cost : $(total_cost(X,offkmeans)) with $(offkmeans.k) clusters")
###### Web-scale k-means clustering #####

web = Webscale()
init!(web,X,k)
i = 1
T=10000
onepass=true
d = zeros(Int64,b)
if onepass
    while (i+b) < n
        update!(web,X[i:(i+b),:])
        i += b
    end
else
    while i < T
        samples = sample(1:n,b,replace=false)
        update!(web,X[samples,:])
        i+=1
    end
end
plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p2 = plot!(web.centers[:,1],web.centers[:,2],t=:scatter,lab="",marker=:o,title="Webscale")
println("Webscale cost : $(total_cost(X,web)) with $(web.k) clusters")
###### An algorithm for Online K Means clustering
### Semi online
q=0
w=fullkmeans.totalcost*(1.0-abs(randn()))
f_c=w/(k*log(n))
C = reshape(X[1,:],1,dim)
for i in 2:n
    val = find_nearest_center(X[i,:],C)[2]
    if val>(f_c*rand())
        C = vcat(C,X[i,:]')
        q += 1
    end
    if q >= 3*k*(1+log(n))
        q = 0
        f_c*=2
    end
end
plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p3 = plot!(C[:,1],C[:,2],t=:scatter,lab="",marker=:o,title="Semi-Online")
println("Semi online cost $(total_cost(X,C)) with $(size(C,1)) clusters")
##### The pragmatic online algorithm
stream = StreamOnline()
init!(stream,X,k)

for i in (k+1):2:n
    update!(stream,X[i:i+1,:])
end
plot(X[:,1],X[:,2],t=:scatter,lab="",alpha=0.3,markerstrokewidth=0)
p4 = plot!(stream.centers[:,1],stream.centers[:,2],t=:scatter,lab="",marker=:o,title="Pragmatic Online")
println("Pragmatic Online cost $(total_cost(X,stream.centers)) with $(stream.k) clusters")
display(plot(p1,p2,p3,p4))
