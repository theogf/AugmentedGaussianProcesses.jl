y = rand(1:10, 10)
l = PoissonLikelihood()
@test AGP.treat_labels!(y, l) == (y, 1, l)
@test_throws AssertionError AGP.treat_labels!(rand(10), l)
