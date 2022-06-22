begin
    import GLPK, JuMP

    function execute_cobra(model_path)
        model = JuMP.read_from_file(model_path)    
        JuMP.set_optimizer(model, GLPK.Optimizer)
        JuMP.optimize!(model)
    end

    print(execute_cobra("test.lp"))
    
end
