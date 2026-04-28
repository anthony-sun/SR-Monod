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

    custom_loss = """
        function expr_loss(ex::AbstractExpression, dataset::Dataset, options)
            q, m, g, h1, h2 =  ex.trees
            is_negative_constant(node) = node.degree == 0 && node.constant && node.val < 0
            num_negative_constants = count(is_negative_constant, get_tree(q)) + count(is_negative_constant, get_tree(m)) + count(is_negative_constant, get_tree(g)) + count(is_negative_constant, get_tree(h1)) + count(is_negative_constant, get_tree(h2))

            if num_negative_constants > 0
                return 1000 * num_negative_constants
            end

            g1_power_constraint(node) = node.degree == 2 && node.l.constant == false && node.l.feature == 1 && node.r.constant == true && node.op == 4
            num_powers_on_c1f = count(g1_power_constraint, get_tree(g))
            if num_powers_on_c1f > 0
                return 1000 * num_powers_on_c1f
            end

            g2_power_constraint(node) = node.degree == 2 && node.l.constant == false && node.l.feature == 2 && node.r.constant == true && node.op == 4
            num_powers_on_c2f = count(g2_power_constraint, get_tree(g))
            if num_powers_on_c2f > 0
                return 1000 * num_powers_on_c2f
            end

            output, completed = eval_tree_array(ex, dataset.X, options)
            !completed && return Inf

            variance = sum((dataset.y .- (sum(dataset.y) / length(dataset.y))) .^ 2) / (length(dataset.y) - 1)

            return (sum((output .- dataset.y) .^ 2) / length(dataset.y)) / variance
        end
        """

    # Function template
    template = TemplateExpressionSpec("(q() / (q() + pow(U, m()))) * g(C1*h1(F), C2*h2(F))",
                                      expressions=["q", "m", "g", "h1", "h2"],
                                      variable_names=["F", "C1", "C2", "U"],
    )

    now = datetime.now()
    # run_id = now.strftime("%d%m%Y-%H%M%S") + "-" + "_".join(data_file.split("/")[-1].split(".")[0].split("_")[:-1])
    run_id = now.strftime("%d%m%Y-%H%M%S") + "-" + data_file.split("/")[-1].split(".")[0].split("_")[0]

    model = PySRRegressor(
        expression_spec=template,                  # Function template specification
        binary_operators=["+", "*", "/", "pow"],   # Operation library
        maxsize=60,                                # Maximum complexity of result equations
        nested_constraints={"pow": {"pow": 0}},    # No complex expressions in powers
        constraints={"pow": (2, 1)},               # Together with complexity of variables = 2 below, this constrains powers to only allow constants (no variables)
        complexity_of_variables=2,                 # Complexity of variables (2 to prevent variables appearing in powers)
        niterations=2000,                          # Number of iterations
        output_directory=output_directory,         # Output directory
        run_id=run_id,                             # Results are saved with this ID
        loss_function_expression=custom_loss       # Custom expression-level loss
    )

    X = data[["F", "C1", "C2", "U"]]  # Selected predictor variables (input dataframe column names)
    y = data["rho"]       # Response variable

    model.fit(X, y)
    print(model)

if __name__ == "__main__":
    main(sys.argv[1:])
