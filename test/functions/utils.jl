@testset "utils/utils.jl" begin
    @test AGP.jitt isa AGP.Jittering
    @test Float64(AGP.jitt) ≈ 1e-4
    @test Float32(AGP.jitt) ≈ 1e-3
    @test Float16(AGP.jitt) ≈ 1e-2

    @test AGP.δ(0, 1) == 0.0
    @test AGP.δ(1, 1) == 1.0

    A = rand(2, 2)
    B = rand(2, 2)
    x = rand(2)
    @test AGP.hadamard(A, B) == A .* B

    X = copy(A)
    AGP.add_transpose!(X)
    @test X == A + A'

    D = A * A' + I
    C = cholesky(D)
    @test AGP.invquad(C, x) ≈ dot(x, D \ x)

    @test AGP.trace_ABt(A, B) ≈ tr(A * B')
    @test AGP.diag_ABt(A, B) ≈ diag(A * B')
    @test AGP.diagv_B(x, B) ≈ Diagonal(x) * B
    @test AGP.κdiagθκ(A, x) ≈ A' * Diagonal(x) * A
    @test AGP.ρκdiagθκ(2.0, A, x) ≈ 2.0 * A' * Diagonal(x) * A
    @test AGP.opt_add_diag_mat(x, A) ≈ A + Diagonal(x)
    @test AGP.safe_expcosh(2.0, 1.0) ≈ exp(2.0) / cosh(1.0)
    @test AGP.logcosh(2.0) ≈ log(cosh(2.0))

    #TODO test 
    #=
    function symcat(S::Symmetric, v::AbstractVector, vv::Real)
        S = vcat(S, v')
        S = hcat(S, vcat(v, vv))
        return Symmetric(S)
    end

    function make_grid(range1, range2)
        return hcat(
            [i for i in range1, j in range2][:],
            [j for i in range1, j in range2][:],
        )
    end
    =#
    @test (2.0 * C).L ≈ cholesky(2.0 * D).L
    @test C * x ≈ D * x
    @test (C + 2.0 * I).L ≈ cholesky(D + 2.0 * I).L
end
