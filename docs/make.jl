using Documenter, Literate
using AugmentedGaussianProcesses

# Create the notebooks and examples markdown using Literate

EXAMPLES = joinpath(@__DIR__, "examples")
MD_OUTPUT = joinpath(@__DIR__, "src", "examples")
NB_OUTPUT = joinpath(@__DIR__, "src", "examples", "notebooks")

ispath(MD_OUTPUT) && rm(MD_OUTPUT; recursive=true)
ispath(NB_OUTPUT) && rm(NB_OUTPUT; recursive=true)

# add links to binder and nbviewer below the first heading of level 1
function preprocess(content)
    sub = s"""
\0
#
# [![](https://img.shields.io/badge/show-nbviewer-579ACA.svg)](@__NBVIEWER_ROOT_URL__/examples/notebooks/@__NAME__.ipynb)
"""
    return replace(content, r"^# # .*$"m => sub; count=1)
end

for file in readdir(EXAMPLES; join=true)
    endswith(file, ".jl") || continue
    Literate.markdown(file, MD_OUTPUT; documenter=true, preprocess=preprocess)
    Literate.notebook(file, NB_OUTPUT; documenter=true)
end

# Make the docs

makedocs(;
    modules=[AugmentedGaussianProcesses],
    format=Documenter.Writers.HTMLWriter.HTML(;
        assets=["assets/icon.ico"], analytics="UA-129106538-2"
    ),
    sitename="AugmentedGaussianProcesses",
    authors="Théo Galy-Fajou",
    pages=[
        "Home" => "index.md",
        "Background" => "background.md",
        "User Guide" => "userguide.md",
        "Kernels" => "kernel.md",
        "Examples" =>
            joinpath.("examples", filter(x -> endswith(x, ".md"), readdir(MD_OUTPUT))),
        "Julia GP Packages" => "comparison.md",
        "API" => "api.md",
    ],
)

# Deploy the docs

deploydocs(;
    deps=Deps.pip("mkdocs", "python-markdown-math"),
    repo="github.com/theogf/AugmentedGaussianProcesses.jl.git",
    target="build",
    push_preview=true,
)
