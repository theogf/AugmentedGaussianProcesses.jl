"""
    Verify that the data is self-consistent and consistent with the likelihood
"""
function check_data!(X::AbstractArray{T₁,N₁},y::AbstractArray{T₂,N₂},likelihood::Likelihood) where {T₁<:Real,T₂,N₁,N₂}
    @assert (size(y,1)==size(X,1)) "There is not the same number of samples in X and y";
    @assert N₁ <= 2 "The input matrix X can only be a vector or a matrix"
    y,nLatent,likelihood = treat_labels!(y,likelihood)
    if N₁ == 1 #Reshape a X vector as a matrix
        return reshape(X,:,1),y,nLatent,likelihood
    else
        return X,y,nLatent,likelihood
    end
end

"""
    Verify that the likelihood and inference are compatible (are implemented)
"""
function check_implementation(model::Symbol,likelihood::L,inference::I) where {I<:Inference,L<:Likelihood}
    if isa(likelihood,GaussianLikelihood)
        if model == :GP && inference isa Analytic
            return true
        elseif (model == :SVGP || model == :VStP) && inference isa AnalyticVI
            return true
        else
            return false
        end
    elseif likelihood isa StudentTLikelihood
        if inference isa AnalyticVI || inference isa QuadratureVI
            return true
        elseif model == :VGP && inference isa GibbsSampling
            return true
        else
            return false
        end
    elseif likelihood isa LaplaceLikelihood
        if inference isa AnalyticVI || inference isa QuadratureVI
            return true
        elseif model == :VGP && inference isa GibbsSampling
            return true
        else
            return false
        end
    elseif likelihood isa HeteroscedasticLikelihood
        if inference isa AnalyticVI
            return true
        else
            return false
        end
    elseif likelihood isa HeteroscedasticLikelihood
        if inference isa AnalyticVI || inference isa QuadratureVI
            return true
        elseif model == :VGP && inference isa GibbsSampling
            return true
        else
            return false
        end
    elseif likelihood isa LogisticLikelihood
        if inference isa AnalyticVI || inference isa QuadratureVI
            return true
        elseif model == :VGP && inference isa GibbsSampling
            return true
        else
            return false
        end
    elseif likelihood isa BayesianSVM
        if inference isa AnalyticVI
            return true
        else
            return false
        end
    elseif likelihood isa SoftMaxLikelihood
        if inference isa MCIntegrationVI
            return true
        else
            return false
        end
    elseif likelihood isa LogisticSoftMaxLikelihood
        if inference isa AnalyticVI
            return true
        elseif model == :VGP && inference isa GibbsSampling
            return true
        elseif inference isa MCIntegrationVI
            return true
        else
            return false
        end
    elseif likelihood isa PoissonLikelihood
        if inference isa AnalyticVI
            return true
        else
            return false
        end
    else
        return true
    end
end
