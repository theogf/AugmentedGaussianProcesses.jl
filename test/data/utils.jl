@testset "data/utils" begin
    X = rand(5, 2)
    wX, T = AGP.wrap_X(X)
    y = rand(2)
end