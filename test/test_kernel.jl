include("../src/AugmentedGaussianProcesses.jl")
import AugmentedGaussianProcesses.KernelFunctions

X1=rand(3);X2=rand(3);
a = AugmentedGaussianProcesses.RBFKernel(1.0);AugmentedGaussianProcesses.compute(a,X1,X2)
b = AugmentedGaussianProcesses.LaplaceKernel(1.0);AugmentedGaussianProcesses.compute(b,X1,X2)
c = AugmentedGaussianProcesses.SigmoidKernel(); AugmentedGaussianProcesses.compute(c,X1,X2)
d = AugmentedGaussianProcesses.PolynomialKernel(); AugmentedGaussianProcesses.compute(d,X1,X1)
d.param[1]
e = AugmentedGaussianProcesses.ARDKernel([1.0,1.0,3.0]); AugmentedGaussianProcesses.compute(e,X1,X2)
f = a+b; AugmentedGaussianProcesses.compute(f,X1,X2)
g = a*b; AugmentedGaussianProcesses.compute(g,X1,X2)
h = c*(d+e); AugmentedGaussianProcesses.compute(g,X1,X2)
i = AugmentedGaussianProcesses.Matern3_2Kernel(); AugmentedGaussianProcesses.compute(i,X1,X2)
j = AugmentedGaussianProcesses.Matern5_2Kernel(); AugmentedGaussianProcesses.compute(j,X1,X2)
using Plots;
gr()
function plotkernel(kernel::AugmentedGaussianProcesses.Kernel;range=[-1.5,1.5],npoints::Int64=100)
    if kernel.distance == AugmentedGaussianProcesses.InnerProduct
        X1 = ones(npoints);
        X2 = collect(range(range[1],range[2],npoints));
        value = zeros(npoints);
        for i in 1:npoints
            value[i] = AugmentedGaussianProcesses.compute(kernel,X1[i],X2[i])
        end
        plot(X2,value,lab="k(x)",xlabel="x")
    elseif kernel.distance == AugmentedGaussianProcesses.SquaredEuclidean
        X1 = zeros(npoints);
        X2 = collect(range(range[1],range[2],npoints));
        value = zeros(npoints);
        for i in 1:npoints
            value[i] = AugmentedGaussianProcesses.compute(kernel,X1[i],X2[i])
        end
        plot(X2,value,lab="k(x)",xlabel="x")
    elseif kernel.distance == AugmentedGaussianProcesses.Identity
        plotlyjs()
        x = collect(range(range[1],range[2],npoints));
        value = broadcast((x,y)->AugmentedGaussianProcesses.compute(kernel,x,y),[i for i in x, j in x],[j for i in x, j in x])
        display(plot(x,x,value,t=:contour,fill=true,cbar=true,xlabel="X",ylabel="Y",title="k(X,Y)"))
    end
end

plotkernel(d)
plotkernel(j)
