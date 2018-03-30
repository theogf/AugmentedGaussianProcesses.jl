import OMGP
using Distributions

minx=-5.0
maxx=5.0

X = (rand(10,2)*10.0)-5.0
y = rand(Uniform(1,4),10)

function y_mapping(y)
    [sum(y .== i) for i in unique(y)]
end

y_mapping(y)
