abstract type GP{L<:Likelihood,I<:Inference,T<:Real,A} end


"""Basic displaying function"""
function Base.show(io::IO,model::GP{<:Likelihood,<:Inference,T}) where T
    print(io,"$(model.Name){$T} model")
end
