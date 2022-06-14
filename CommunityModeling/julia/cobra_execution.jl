# using Pkg
import GLPK, MathProgBase, JuMP, MathOptInterface #, COBRA
# include("$(Pkg.dir("COBRA"))/src/connect.jl")

mutable struct SolverConfig
    name      ::String
    handle
end

function build_LP(model)
    for i = 1:length(model.csense)
        if model.csense[i] == 'E'  model.csense[i] = '=' end
        if model.csense[i] == 'G'  model.csense[i] = '>' end
        if model.csense[i] == 'L'  model.csense[i] = '<' end
    end

    return MathProgBase.HighLevelInterface.buildlp(model.osense * model.c, model.S, model.csense, model.b, model.lb, model.ub, GLPK.Optimizer)
end

function execute_cobra(model_path)
    # include("$(Pkg.dir("COBRA"))/config/solverCfg.jl")
    # model = COBRA.loadModel(model_path, "S", "CMM_iAH991V2_iML1515.kb")
    model = JuMP.read_from_file("test.lp")    
    JuMP.set_optimizer(model, GLPK.Optimizer)
    # solutionLP = MathProgBase.HighLevelInterface.solvelp(build_LP(model))
    JuMP.optimize!(model)

    # adapt the objective value
    # if solutionLP.status == :Optimal
    #     solutionLP.objval = model.osense * solutionLP.objval
    # end

    return solutionLP
end