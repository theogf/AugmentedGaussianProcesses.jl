##Work in progress to test methods##


mutable struct DataSelection <: KMeansAlg
    accepting_function::Function ###From given parameters return if point is accepted (return true or false)
    lim
    k::Int64
    centers::Array{Float64,2}
    function DataSelection(;func=RandomAccept_Mean,lim=[0.9,0.4])
        return new(func,lim)
    end
end

function init!(alg::DataSelection,X,y,model,k::Int64)
    n = size(X,1)
    # init_k = max(1,ceil(Int64,n/10))
    init_k = min(10,n)
    alg.centers = reshape(X[sample(1:n,init_k,replace=false),:],init_k,size(X,2))
    alg.k = init_k
end

function update!(alg::DataSelection,X,y,model)
    b = size(X,1)
    new_centers = []
    new_k = 0
    diff = get_likelihood_diffs(model,alg,X)
    for (i,val) in enumerate(diff)
        if val > alg.lim
            new_k += 1
            push!(new_centers,X[i,:]')
        end
    end
    # for i in 1:b
    #     mu,sig = model.fstar(reshape(X[i,:],1,size(X,2)))
    #     # println("σ: $sig,\tabs(mu-f): $(1-exp(-0.5*(mu-y[i])[1]^2/sig[1])),\tquant: $(log(sqrt(sig))+0.5*(y[i]-mu[1])^2/sig[1])")
    #     if alg.accepting_function(alg,model,mu,sig,X[i,:],y[i])
    #         # μ_new,σ_new = model.predictproba(X[i,:])
    #         # alg.centers = vcat(alg.centers,X[i,:]')
    #         # update_model!(model,reshape(X[i,:],1,size(X,2)),μ_new,σ_new)
    #         # alg.k += 1
    #         new_k += 1
    #         push!(new_centers,X[i,:]')
    #     end
    # end
    for v in new_centers
        alg.centers = vcat(alg.centers,v)
    end
    alg.k += new_k
end

function RandomAccept_Mean(alg,model,mu,sig,X,y)
    diff_f = 1.0.-exp.(-0.5*(mu.-y)[1]^2/sig[1])
    d = find_nearest_center(X,alg.centers,model.kernel)[2]
    if d>2*(1-alg.lim[1])
        # println("Distance point")
        return true
    elseif diff_f > alg.lim[2] && d<2*(1-alg.lim[1]-0.05)
        # println("Likelihood point")
        return true
    end
    return false
    # println(sig[1])
    #return sig[1]>0.8*(1-diff_f)
    # return KLGP(mu[1],sig[1],y,0.001)>10
    # return (d>(1-2*alg.lim[1]) || diff_f>0.5)
    # return JSGP(mu[1],sig[1],y,0.001)>10
    # return sig[1]>0.8
    # return (0.5*sqrt(sig[1])+0.5*diff_f)>rand()
    # return 1.0*sqrt(sig[1])>rand()
end

function get_likelihood_diffs(model,alg,new_points)
    N_new = size(new_points,1)
    # old_ind = copy(model.MBIndices)
    # model.MBIndices = collect(1:model.nSamples)
    # model.indpoints_updated = true
    # computeMatrices!(model)
    s = 1.0#model.StochCoeff
    k_u = kernelmatrix(alg.centers,new_points,model.kernel)
    a = model.invKmm*k_u
    c = kerneldiagmatrix(new_points,model.kernel) - diag(k_u'*a)
    kfu = kernelmatrix(model.X[model.MBIndices,:],new_points,model.kernel)
    # println("c: $c")

    b = [1.0/sqrt(c[i]).*(model.Knm*model.invKmm*k_u[:,i]-kfu[:,i]) for i in 1:N_new]
    A = inv(model.gnoise*I + model.κ*model.Knm')
    v = [1+s*dot(b[i],A*b[i]) for i in 1:N_new]
    diffL = zeros(N_new)
    diffL .+= log.(v)
    # println("LogV = $(-0.5*diffL)")
    diffL .+= - 1.0/model.gnoise*[tr(b[i]*b[i]') for i in 1:N_new]
    # println("LogV+tr = $(-0.5*diffL)")
    diffL .+= - [dot(model.y[model.MBIndices],(A*b[i]*b[i]'*A)/v[i]*model.y[model.MBIndices]) for i in 1:N_new]
    # model.MBIndices = copy(old_ind)
    # model.indpoints_updated = true
    # computeMatrices!(model)
    println("Diffl = $(-0.5*diffL)")
    # return ifelse(c.<model.gnoise,0.1,-0.5*s*diffL)
    return -0.5.*diffL
end


"""BBLAH"""
function get_likelihood_diff(model,alg,new_point)
    old_ind = copy(model.MBIndices)
    model.MBIndices = collect(1:model.nSamples)
    model.indpoints_updated = true
    computeMatrices!(model)
    k_u = vec(kernelmatrix(alg.centers,reshape(new_point,1,size(new_point,1)),model.kernel))
    a = model.invKmm*k_u
    c = kerneldiagmatrix(new_point,model.kernel)[1] - dot(k_u,a)
    # if c < 0.001
    #     model.MBIndices = copy(old_ind)
    #     model.indpoints_updated = true
    #     computeMatrices!(model)
    #     return 0.0001
    # end
    println("c: $c")
    kfu = kernelmatrix(model.X,reshape(new_point,1,size(new_point,1)),model.kernel)
    # kfu = kernelmatrix(model.X[model.MBIndices,:],new_point,model.kernel)
    b = 1.0/sqrt(c)*(model.Knm*model.invKmm*k_u-kfu)
    A = inv(model.gnoise*I + model.κ*model.Knm')
    # App = (1/model.gnoise*I-(model.StochCoeff/model.gnoise^2)*model.κ*model.Σ*model.κ')
    # println("b : $b")
    # A = (1/model.gnoise*I-model.StochCoeff*(1.0/model.gnoise^2)*model.κ*inv(model.invKmm+model.StochCoeff*1/model.gnoise*model.κ'*model.κ)*model.κ')
    # v = 1+model.StochCoeff*dot(b,A*b)
    v = 1+dot(b,A*b)
    diffL = 0.0
    diffL += log(v)
    diffL += - 1.0/model.gnoise*tr(b*b')

    # diffL += - model.StochCoeff*1.0/model.gnoise*tr(b*b')
    diffL += - dot(model.y[model.MBIndices],(A*b*b'*A)/v*model.y[model.MBIndices])
     # diffL+= - model.StochCoeff^2*dot(model.y[model.MBIndices],(A*b*b'*A)/v*model.y[model.MBIndices])
     model.MBIndices = copy(old_ind)
     model.indpoints_updated = true
     computeMatrices!(model)
    return -0.5*diffL
end


export LikelihoodImprovement, get_likelihood_diff
function LikelihoodImprovement(alg,model,mu,sig,X,y)
    d = get_likelihood_diff(model,alg,X)
    # println("$d , $(d/abs(ELBO(model)))")
    if get_likelihood_diff(model,alg,X) > alg.lim
        return true
    end
    return false
end
