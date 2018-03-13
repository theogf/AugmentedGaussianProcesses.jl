include("KernelFunctions.jl")


a= RBFKernel(3.0)
b= RBFKernel(0.5)
c = a+b
d = a*b
e = a*(c+d)
X1=rand(5)
X2=rand(5)


function foo_gradient(kernel)
    return 0.5*compute_deriv(kernel,[0.5,0.5],[0.2,0.3],true)
end

grad = foo_gradient(a)
apply_gradients!(a,grad,true)

grad = foo_gradient(c)
apply_gradients!(c,grad,true)

grad = foo_gradient(d)
apply_gradients!(d,grad,true)

grad = foo_gradient(e)
apply_gradients!(e,grad,true)

point_grad = compute_point_deriv(a,X1,X2)
point_grad = compute_point_deriv(c,X1,X2)
point_grad = compute_point_deriv(d,X1,X2)
point_grad = compute_point_deriv(e,X1,X2)

rc = compute(c,X1,X2)
ra = compute(a,X1,X2)
rb = compute(b,X1,X2)
diff = rc-(ra+rb)

X = rand(4,4)
Y = rand(4,4)

function the_trace(J)
    return trace(J)
end


function sum_trace(Js)
    return trace(Js[1])+trace(Js[2])+trace(Js[3])
end
J = compute_unmappedJ(c,X,Y)
Js = compute_J(c,J,4,4)
JJs = [Js,2*Js,3*Js]
6*trace(Js[2])
grad = compute_hyperparameter_gradient(c,sum_trace,JJs)
apply_gradients!(c,grad)
