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
