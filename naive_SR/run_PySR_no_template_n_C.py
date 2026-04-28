import sys
import ast
import numpy as np
import pandas as pd
from datetime import datetime
from pysr import PySRRegressor, TemplateExpressionSpec

# PySR version v1.4.0

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
        powers_on_c(node) = node.degree == 2 && node.l.constant == false && node.l.feature == 2 && node.r.constant == true && node.op == 4
        num_powers_on_c = count(powers_on_c, tree)
        if num_powers_on_c > 0
            return 1000 * num_powers_on_c
        end
      
        if num_negative_constants > 0
            return L(1000 * num_negative_constants)
        end
        
        prediction, flag = eval_tree_array(tree, dataset.X, options)
        if !flag
            return L(Inf)
        end

        variance = sum((dataset.y .- (sum(dataset.y) / length(dataset.y))) .^ 2) / (length(dataset.y) - 1)

        return (sum((prediction .- dataset.y) .^ 2) / length(dataset.y)) / variance
    end"""

    now = datetime.now()
    # run_id = now.strftime("%d%m%Y-%H%M%S") + "-" + "_".join(data_file.split("/")[-1].split(".")[0].split("_")[:-1])
    run_id = now.strftime("%d%m%Y-%H%M%S") + "-" + data_file.split("/")[-1].split(".")[0].split("_")[0]

    model = PySRRegressor(
        binary_operators=["+", "*", "/", "pow"],   # Operation library
        maxsize=40,                                # Maximum complexity of result equations
        nested_constraints={"pow": {"pow": 0}},    # No complex expressions in powers
        constraints={"pow": (2, 1)},               # Together with complexity of variables = 2 below, this constrains powers to only allow constants (no variables)
        complexity_of_variables=2,                 # Complexity of variables (2 to prevent variables appearing in powers)
        niterations=2000,                          # Number of iterations
        output_directory=output_directory,         # Output directory
        run_id=run_id,                             # Results are saved with this ID
        loss_function=custom_loss,                 # Custom expression-level loss
    )

    # Train-test split
    train_data = data[data["C"].isin(data["C"].unique()[::2])]
    test_data = data[data["C"].isin(data["C"].unique()[1::2])]

    X = train_data[["n", "C"]]  # Selected predictor variables (input dataframe column names)
    y = train_data["rho"]       # Response variable

    model.fit(X, y)
    print(model)

    train_data.to_csv(output_directory + run_id + "_train_data.csv", index=False)
    test_data.to_csv(output_directory + run_id + "_test_data.csv", index=False)

if __name__ == "__main__":
    main(sys.argv[1:])
