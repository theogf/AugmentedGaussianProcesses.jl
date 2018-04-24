include("../src/OMGP.jl")
import OMGP.KernelFunctions

X1=rand(3);X2=rand(3);
a = OMGP.RBFKernel(1.0);OMGP.compute(a,X1,X2)
b = OMGP.LaplaceKernel(1.0);OMGP.compute(b,X1,X2)
c = OMGP.SigmoidKernel(); OMGP.compute(c,X1,X2)
d = OMGP.PolynomialKernel(); OMGP.compute(d,X1,X1)
d.param[1]
e = OMGP.ARDKernel([1.0,1.0,3.0]); OMGP.compute(e,X1,X2)
f = a+b; OMGP.compute(f,X1,X2)
g = a*b; OMGP.compute(g,X1,X2)
h = c*(d+e); OMGP.compute(g,X1,X2)
i = OMGP.Matern3_2(); OMGP.compute(i,X1,X2)
j = OMGP.Matern5_2(); OMGP.compute(j,X1,X2)
using Plots;
pyplot();
function plotkernel(kernel::OMGP.Kernel;range=[-1.5,1.5],npoints::Int64=100)
    if kernel.distance == OMGP.InnerProduct
        X1 = ones(npoints);
        X2 = collect(linspace(range[1],range[2],npoints));
        value = zeros(npoints);
        for i in 1:npoints
            value[i] = OMGP.compute(kernel,X1[i],X2[i])
        end
        plot(X2,value,lab="k(x)",xlabel="x")
    elseif kernel.distance == OMGP.SquaredEuclidean
        X1 = zeros(npoints);
        X2 = collect(linspace(range[1],range[2],npoints));
        value = zeros(npoints);
        for i in 1:npoints
            value[i] = OMGP.compute(kernel,X1[i],X2[i])
        end
        plot(X2,value,lab="k(x)",xlabel="x")
    elseif kernel.distance == OMGP.Identity
        plotlyjs()
        x = collect(linspace(range[1],range[2],npoints));
        value = broadcast((x,y)->OMGP.compute(kernel,x,y),[i for i in x, j in x],[j for i in x, j in x])
        display(plot(x,x,value,t=:contour,fill=true,cbar=true,xlabel="X",ylabel="Y",title="k(X,Y)"))
    end
end

plotkernel(d)
plotkernel(j)
