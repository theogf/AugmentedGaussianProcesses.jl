include("../src/polyagammasampler.jl")

using PGSampler
using Plots
using StatsBase
plotlyjs()
println("Testing")
p = PolyaGammaDist()
N=100000; v= zeros(N);w=zeros(N);u=zeros(N);
for i in 1:N
  v[i] = p.draw(1.0,0.0);
  w[i] = p.draw(2.0,0.0);
  u[i] = p.draw(3.0,0.0);
end
# figure(1);clf();
hv = fit(Histogram,v,nbins=floor(Int64,N/100),closed=:left)
hw = fit(Histogram,w,nbins=floor(Int64,N/100),closed=:left)
hu = fit(Histogram,u,nbins=floor(Int64,N/100),closed=:left)
plot(hv.edges,hv.weights,lab="b=1,c=0")
plot!(hw.edges,hw.weights,lab="b=2,c=0")
p1 = plot!(hu.edges,hu.weights,lab="b=3,c=0")
println("Testing 2")
for i in 1:N
  v[i] = p.draw(1.0,0.0);
  w[i] = p.draw(1.0,2.0);
  u[i] = p.draw(1.0,4.0);
end
bins = linspace(0.0,1.0,floor(Int64,N/500))
hv = fit(Histogram,v,bins,closed=:left)
hw = fit(Histogram,w,bins,closed=:left)
hu = fit(Histogram,u,bins,closed=:left)
plot(hv.edges,hv.weights,lab="b=1,c=0")
plot!(hw.edges,hw.weights,lab="b=1,c=2")
p2 = plot!(hu.edges,hu.weights,lab="b=1,c=4")
plot(p1,p2,layout=(1,2))
