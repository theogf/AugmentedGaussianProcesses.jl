# import OMGP
using Distributions
using StatsBase
using Gallium
N_data = 10
N_class = 4
N_test = 20
minx=-5.0
maxx=5.0


X = (rand(N_data,2)*(maxx-minx))+minx
y = rand(DiscreteUniform(1,N_class),N_data)
OMGP.MultiClass(X,y)
