cd(dirname(@__FILE__))
using BenchmarkTools, DelimitedFiles, Statistics


function load_results(benchmarkname::String;version1=nothing,version2=nothing)
    files = readdir("results")
    files = files[occursin.([benchmarkname],files)]
    print("Comparing file ")
    if version1 == nothing
        prev_result = BenchmarkTools.load("results/"*files[end-1])[1]
        print(files[end-1])
    else
        prev_result = BenchmarkTools.load("results/"*benchmarkname*version1)[1]
        print(benchmarkname*version1*"\n")
    end
    print(" and ")
    if version2 == nothing
        last_result = BenchmarkTools.load("results/"*files[end])[1]
        print(files[end]*"\n")
    else
        last_result = BenchmarkTools.load("results/"*benchmarkname*version2)[1]
        print(benchmarkname*version2*"\n")
    end



    return prev_result,last_result
end


function compare_versions(benchmarkname::String;version1=nothing,version2=nothing)
    r2,r1 = load_results(benchmarkname,version1=version1,version2=version2)
    for k in keys(r2)
        println(k)
        for v in keys(r2[k])
            if haskey(r1[k],v)
                change = judge(median(r1[k][v]),median(r2[k][v]))
                print("For model ")
                printstyled("$k",color=:red)
                print(", and test ")
                printstyled("$v :\n",bold=true,color=:blue)
                @show change
            end
        end
    end
end
