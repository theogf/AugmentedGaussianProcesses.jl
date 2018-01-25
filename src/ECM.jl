if !isdefined(:KernelFunctions); include("KernelFunctions.jl"); end;

module ECM

using Distributions
using KernelFunctions

export ECMTraining, PredicECM, PredictProbaECM


mutable struct ECMBSVM

end


function ECMTraining(X::Array{Float64,2}, y::Array{Float64,1}; kernel=0,γ::Float64=1.0, nepochs::Int64=100,ϵ::Float64 = 1e-5,Θ=[1.0],verbose=false)
  #initialization of parameters
    n = size(X,1)
    k = size(X,1)
    Y = Diagonal(y)
    f = randn(k)
    invλ = abs.(rand(k))
    if kernel == 0
      kernel = Kernel("rbf",Θ[1],params=Θ[2])
    elseif typeof(kernel)==AbstractString
      if length(Θ)>1
        kernel = Kernel(kernel,Θ[1],params=Θ[2])
      else
        kernel = Kernel(kernel,Θ[1])
      end
    end
    #Initialize a vector using the prior information
    K = CreateKernelMatrix(X,kernel.compute)
    i = 1
    diff = Inf;
    while i < nepochs && diff > ϵ
        prev_λ = 1./invλ; prev_f = f;
        #Expectation Step
        invλ = sqrt(1.0+2.0/γ)./(abs.(1.0-y.*f))
        #Maximization Step
        f = K*inv((K+1.0/γ*diagm(1./invλ)))*Y*(1+1./invλ)
        diff = norm(f-prev_f);
        i += 1
        if verbose
          println("$i : diff = $diff")
        end
    end
    if verbose
      println("Henao stopped after $i iterations")
    end
    return (invλ,K,kernel,y,f)
end

function PredicECM(X,y,X_test,invλ,K,γ,kernel)
  n = size(X,1)
  n_t = size(X_test,1)
  predic = zeros(n_t)
  Σ = inv(K+1/γ*diagm(1./invλ))
  top = Σ*diagm(y)*(1+1./invλ)
  for i in 1:n_t
    k_star = zeros(n)
    for j in 1:n
      k_star[j] = kernel.compute(X[j,:],X_test[i,:])
    end
    predic[i] = dot(k_star,top)
  end
  return predic
end

function PredictProbaECM(X,y,X_test,invλ,K,γ,kernel)
  n = size(X,1)
  n_t = size(X_test,1)
  predic = zeros(n_t)
  Σ = inv(K+1/γ*diagm(1./invλ))
  top = Σ*diagm(y)*(1+1./invλ)
  for i in 1:n_t
    k_star = zeros(n)
    for j in 1:n
      k_star[j] = kernel.compute(X[j,:],X_test[i,:])
    end
    k_starstar = kernel.compute(X_test[i,:],X_test[i,:])
    predic[i] = cdf(Normal(),dot(k_star,top)/(1+k_starstar-dot(k_star,Σ*k_star)))
  end
  return predic
end
end
