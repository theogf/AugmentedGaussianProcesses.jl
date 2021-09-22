@testset "multiclass" begin
    y = [1, 2, 3, 1, 1, 2, 3]
    l = LogisticSoftMaxLikelihood(3)
    AGP.create_mapping!(l, y)
    @test sort(l.class_mapping) == [1, 2, 3]
    @test l.ind_mapping == Dict(1 => 1, 2 => 2, 3 => 3)
    Y = AGP.create_one_hot(l, y[1:3])
    @test Y == BitMatrix([1 0 0; 0 1 0; 0 0 1])

    l = LogisticSoftMaxLikelihood(2)
    @test_throws ErrorException AGP.create_mapping!(l, y)

    y = [1, 2, 1, 1]
    l = LogisticSoftMaxLikelihood(3)
    AGP.create_mapping!(l, y)
    @test l.class_mapping == [1, 2, 3]
    @test l.ind_mapping == Dict(1 => 1, 2 => 2, 3 => 3)
    Y = AGP.create_one_hot(l, y)
    @test Y == BitMatrix([1 0 0; 0 1 0; 1 0 0; 1 0 0])
    @test AGP.n_latent(l) == 3

    y = ["b", "a", "c", "a", "a"]
    l = LogisticSoftMaxLikelihood(3)
    AGP.create_mapping!(l, y)
    @test l.class_mapping == ["b", "a", "c"]
    @test l.ind_mapping == Dict("b" => 1, "a" => 2, "c" => 3)
    Y = AGP.create_one_hot(l, y)
    @test Y == BitMatrix([1 0 0; 0 1 0; 0 0 1; 0 1 0; 0 1 0])

    l = LogisticSoftMaxLikelihood(["a", "b", "c"])
    @test l.ind_mapping == Dict("a" => 1, "b" => 2, "c" => 3)
    Y = AGP.create_one_hot(l, y)
    @test Y == BitMatrix([0 1 0; 1 0 0; 0 0 1; 1 0 0; 1 0 0])

    l = LogisticSoftMaxLikelihood(3)
    Y = AGP.treat_labels!(y, l)
    @test l.class_mapping == ["b", "a", "c"]
    @test l.ind_mapping == Dict("b" => 1, "a" => 2, "c" => 3)
    @test Y == BitMatrix([1 0 0; 0 1 0; 0 0 1; 0 1 0; 0 1 0])
end
