@testset "datacontainer" begin
    n_samples = 10
    n_dim = 4
    n_out = 3
    X = [rand(n_dim) for _ in 1:n_samples]
    y = rand(n_samples)
    ys = [rand(n_samples) for _ in 1:n_out]
    ybad = rand(n_samples + 1)
    ysbad = vcat(ys, ybad)
    @testset "DataContainer" begin
        data = AGP.wrap_data(X, y)
        @test AGP.n_sample(data) == n_samples
        @test AGP.n_dim(data) == n_dim
        @test AGP.n_output(data) == 1
        @test AGP.input(data) == X
        @test AGP.output(data) == y
        @test_throws ErrorException AGP.wrap_data(X, ybad)
        # Multiple ys case
        data = AGP.wrap_data(X, ys)
        @test AGP.n_sample(data) == n_samples
        @test AGP.n_dim(data) == n_dim
        @test AGP.n_output(data) == 1
        @test AGP.input(data) == X
        @test AGP.output(data) == ys
        @test_throws ErrorException AGP.wrap_data(X, ysbad)
    end

    @testset "MODataContainer" begin
        data = AGP.wrap_modata(X, ys)
        @test AGP.n_sample(data) == n_samples
        @test AGP.n_dim(data) == n_dim
        @test AGP.n_output(data) == n_out
        @test AGP.input(data) == X
        @test AGP.output(data) == ys
        @test_throws ErrorException AGP.wrap_data(X, ysbad)
    end

    X = rand(5, 2)
    @testset "wrap_X" begin
        @test AGP.wrap_X(X, 1)[1] isa AbstractVector
        @test AGP.wrap_X(X, 2)[1] isa AbstractVector
        @test AGP.wrap_X(vec(X))[1] isa AbstractVector
        @test AGP.wrap_X(vec(X))[2] == Float64
    end
end
