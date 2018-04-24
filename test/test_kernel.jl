using OMGP.KernelFunctions

X1=rand(3);X2=rand(3;
a = RBFKernel(1.0);compute(a,X1,X2)
b = LaplaceKernel(1.0);compute(b,X1,X2)
c = SigmoidKernel(); compute(c,X1,X2)
d = PolynomialKernel(); compute(d,X1,X1)
e = ARDKernel([1.0,1.0,3.0]); compute(e,X1,X2)
f = a+b; compute(f,X1,X2)
g = a*b; compute(g,X1,X2)
h = c*(d+e); compute(g,X1,X2)

using Plots;
pyplot();


function plotkernel(kernel::Kernel;range=[-3.0,3.0],npoints::Int64=100)
    if kernel.pairwisefunction == InnerProduct
        X1 = ones(npoints);
        X2 = collect(linspace(range[1],range[2],npoints));
        value = zeros(npoints);
        for i in 1:npoints
            value[i] = compute(kernel,X1[i],X2[i])
        end
        plot(X2,value,lab="k(x)",xlabel="x")
    elseif kernel.pairwisefunction == SquaredEuclidean
        X1 = zeros(npoints);
        X2 = collect(linspace(range[1],range[2],npoints));
        value = zeros(npoints);
        for i in 1:npoints
            value[i] = compute(kernel,X1[i],X2[i])
        end
        plot(X2,value,lab="k(x)",xlabel="x")
    elseif kernel.pairwisefunction == Identity
        plotlyjs()
        x = collect(linspace(range[1],range[2],npoints));
        value = broadcast((x,y)->compute(kernel,x,y),[i for i in x, j in x],[j for i in x, j in x])
        display(plot(x,x,value,t=:contour,fill=true,cbar=true,xlabel="X",ylabel="Y",title="k(X,Y)"))
    end
end
