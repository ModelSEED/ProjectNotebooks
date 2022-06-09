using COBRA
using Pkg
include("$(Pkg.dir("COBRA"))/config/solverCfg.jl")
# include("$(Pkg.dir("COBRA"))/src/connect.jl")

function execute_cobra(model_path)
    solverName = :GLPKMathProgInterface
    solver = changeCobraSolver(solverName, solParams)
    model = loadModel(model_path, "S", "CMM_iAH991V2_iML1515.kb")
    return solveCobraLP(model, solver)
end