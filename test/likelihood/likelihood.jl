@testset "likelihood" begin
    struct NewLikelihood <: AGP.AbstractLikelihood end
    @test length(NewLikelihood()) == 1
end
