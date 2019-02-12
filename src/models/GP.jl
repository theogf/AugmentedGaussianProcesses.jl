abstract type GP{L<:Likelihood,I<:Inference,T<:Real,A} end


"""Basic displaying function"""
function Base.show(io::IO,model::VGP{<:Likelihood,<:Inference,T}) where T
    print(io,"Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end


function Base.show(io::IO,model::SVGP{<:Likelihood,<:Inference,T}) where T
    print(io,"Sparse Variational Gaussian Process with a $(model.likelihood) infered by $(model.inference) ")
end
