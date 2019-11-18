struct CustomLikelihood{T} <: Likelihood{T}
    pdf::Function
    nLatent::Int
end

function treat_labels!(y,likelihood::CustomLikelihood)
    return y,likelihood.nLatent,likelihood
end

pdf(d::CustomLikelihood,y::Real,fs::AbstractVector)=d.pdf(y,fs)
