# SR-Monod
Symbolic regression and Monod dynamics to study bacterial growth

## Data treatment
Use file **2025SRMO_pars.ipynb** to generate random parameters for simulated data.
Use file **2025SRMO_data.ipynb** to clean raw experimental data and to generate simulated data.

## Random Forest
Use file **2025SRMO_rfml.ipynb** to run Random Forest analyses on cleaned data and generate manuscript Fig.2.

## Symbolic regression
File **run_pySR.py** represents our final PySR model, complete with custom loss and specified expression template for prediction of per-capita growth rate using cumulative population size $N_c$ (as $F = e^{-N_c}$), resource concentration $C$ and time $t$ (as $U = e^{-t})$.

### Instructions:

The script takes in two arguments:

1. Input data file -- a .csv file (to be read in as a Pandas dataframe) with appropriately named columns referring to input variables and the response variable, as specified within the script.
2. Output directory for PySR result files.

We use one species' data per run. The runs used 32 computing cluster cores with 8G memory per core and the algorithm is set to run for 2000 iterations, which is achieved in few hours.

See the script for further details.

Use file **2025SRMO_sreq.ipynb** to analyse symbolic regression results and generate manuscript Fig.3.
