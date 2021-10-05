@testset "data/utils" begin
    X = rand(5, 2)
    wX, T = AGP.wrap_X(X)
    y = rand(5)
    ys = [rand(5) for _ in 1:3]
    indices = rand(1:5, 3)
    data = AGP.wrap_data(wX, y)
    @test AGP.view_x(data, indices) == view(wX, indices)
    l = GaussianLikelihood()
    @test AGP.view_y(l, data, indices) == view(y, indices)
    @test AGP.view_y(l, y, indices) == view(y, indices)
    y_softmax = falses(5, 3)
    data = AGP.wrap_data(wX, y_softmax)
    l = SoftMaxLikelihood(3)
    @test AGP.view_y(l, data, indices) == view(y_softmax, indices, :)

    l = GaussianLikelihood()
    data = AGP.wrap_modata(wX, ys)
    @test AGP.view_y(l, data, indices) == view.(ys, Ref(indices))
    @test AGP.view_y((l,), data, indices) == view.(ys, Ref(indices))
end
