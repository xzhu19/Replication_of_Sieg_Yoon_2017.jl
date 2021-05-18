push!(LOAD_PATH,"../src/")
using Documenter, Replication_of_Sieg_Yoon_2017

makedocs(modules = [Replication_of_Sieg_Yoon_2017], sitename = "Replication_of_Sieg_Yoon_2017.jl")

deploydocs(repo = "github.com/vho97/Replication_of_Sieg_Yoon_2017.jl.git", devbranch = "docs")
