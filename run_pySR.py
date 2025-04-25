import sys
import ast
import numpy as np
import pandas as pd
from datetime import datetime
from pysr import PySRRegressor, TemplateExpressionSpec

def main(args):
    """
    Required arguments:
    1. Input data file -- a Pandas dataframe with appropriately named columns
    2. Output directory (will be created by PySR if not existing)
    """

    data_file = args[0]
    output_directory = args[1]
    data = pd.read_csv(data_file)

    custom_loss = """function eval_loss_custom(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    
        # Check how many constants are negative
        is_negative_constant(node) = node.degree == 0 && node.constant && node.val::T < 0
        num_negative_constants = count(is_negative_constant, tree)
      
        # Check if C has powers
      
        if num_negative_constants > 0
            return L(1000 * num_negative_constants)
        end
        
        prediction, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end
        return sum((prediction .- dataset.y) .^ 2) / dataset.n
    end"""

    # Function template
    template = TemplateExpressionSpec(function_symbols=["a", "b", "f"],
                                      combine="((; a, b, f), (F, C, U)) -> (a() / (a() + pow(U, b()))) * f(F, C)",
    )

    now = datetime.now()
    run_id = now.strftime("%d%m%Y-%H%M%S") + "-" + "_".join(data_file.split("/")[-1].split(".")[0].split("_")[:-1])

    model = PySRRegressor(
        expression_spec=template,                  # Function template specification
        binary_operators=["+", "*", "/", "pow"],   # Operation library
        maxsize=30,                                # Maximum complexity of result equations
        nested_constraints={"pow": {"pow": 0}},    # No complex expressions in powers
        constraints={"pow": (2, 1)},               # Together with complexity of variables = 2 below, this constrains powers to only allow constants (no variables)
        complexity_of_variables=2,                 # Complexity of variables
        loss_function=custom_loss,                 # Set custom loss function
        niterations=2000,                          # Number of iterations
        output_directory=output_directory,         # Output directory
        run_id=run_id                              # Results are saved with this ID
    )

    X = data[["F", "C", "U"]]  # Selected predictor variables (input dataframe column names) 
    y = data["rho"]            # Response variable

    model.fit(X, y)
    print(model)

if __name__ == "__main__":
    main(sys.argv[1:])
