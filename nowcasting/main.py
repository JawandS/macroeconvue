# Reference: https://github.com/baptiste-meunier/NowcastingML_3step/blob/main/11-Github/Main.R

"""
Translated from R script by Chinn et al. (2023)

Description:
  This Python script implements a three-step forecasting approach:
    1. Data cleaning and re-alignment
    2. Pre-selection of regressors (using methods such as LARS or SIS)
    3. Factor extraction via PCA (with Kaiser’s criterion: keep factors with eigenvalue > 1)
    4. Regression forecasting (using one or several regression techniques)

Inputs:
  - A data file (Excel) in folder "1-Inputs" that contains:
      • A column 'target'
      • A column 'date' in format YYYY-MM-15
      • Additional regressors (already transformed)
  - User-defined parameters for out-of-sample forecast dates, forecast horizons,
    pre-selection methods, number of variables kept, regression techniques, and
    optional calibration settings.

Outputs:
  - For each combination of parameters, CSV files are written in a subfolder of "2-Output":
      • Predictions (with date, true value, and forecasts)
      • RMSE summaries (for crisis and non-crisis samples)
      • An overall summary combining the RMSE’s
       
Before running this script, please ensure that the functions below are defined (or adapted)
in your Python environment:
    • do_clean (data cleaning; analogous to R's doClean)
    • do_align (re-alignment; analogous to R's doAlign)
    • pre_select (pre-selection; analogous to R's pre_select)
    • run_regressions (runs the chosen regression methods)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import custom functions (assumed to exist in a module, e.g., in a folder "functions")
# These functions are the Python counterparts of the R functions in "3-Functions"
from functions import do_clean           # corresponds to R's doClean
from functions import do_align           # corresponds to R's doAlign
from functions import pre_select      # corresponds to R's pre_select
from functions import run_regressions   # corresponds to R's run_regressions

# ----------------------------------------------------------------------------
# STEP 0 - SET USER PARAMETERS FOR THE HORSERACE
# ----------------------------------------------------------------------------

# Part 1: General settings
name_input = "fred-md.csv"          # Input file (located in "1-Inputs")
min_start_date = "1990-01-01"             # Minimum start date for variables
start_date_oos = "2008-01-01"             # Out-of-sample forecast start date
end_date_oos = "2010-01-01"               # Out-of-sample forecast end date

# Part 2: Forecast horizons, pre-selection, and regression combinations
list_h = [-2, -1]                       # List of forecast horizons (negative: backcast, 0: nowcast, positive: forecast)
list_methods = [1, 2]                   # Pre-selection methods (e.g., 1 = LARS, 2 = SIS, etc.)
list_n = [40, 60]                       # Number of variables kept after pre-selection
list_reg = [1, 7]                       # Regression techniques (e.g., 1 = OLS, 7 = XGBoost linear)

# Part 3: Optional inputs (calibration)
do_factors = 1                          # 1 = extract factors (via PCA), 0 = do not
fast_bma = 1                            # Fast version of Bayesian Model Averaging if selected
n_per = 12                              # Number of periods for hyper-parameter validation
sc_ml = 1                               # 1 = scale variables for machine learning, 0 = no scaling
fast_MRF = 1                            # Fast tuning for macroeconomic random forest

# ----------------------------------------------------------------------------
# STEP 1 - READ (AND CLEAN) THE USER-PROVIDED DATASET
# ----------------------------------------------------------------------------

# Read data from Excel and convert date column to datetime
input_path = os.path.join("1-Inputs", name_input)
data_init = pd.read_csv(input_path)
data_init['date'] = pd.to_datetime(data_init['date'], format='%m-%d-%Y')

# Clean dataset (e.g., interpolate missing values, remove variables starting after min_start_date)
data_rmv = do_clean(data_init, min_start_date)

# ----------------------------------------------------------------------------
# STEP 2 - PERFORM THE HORSERACE (LOOP OVER USER PARAMETERS)
# ----------------------------------------------------------------------------

# Dictionary to collect summary information for all horizons
summary_ps_meth_all = {}

for horizon in list_h:
    # Re-align dataset based on the forecast horizon
    data_real = do_align(data_rmv, horizon)
    
    # Initialize a container for the summary for this horizon
    summary_ps_meth = {}
    
    # Loop over pre-selection methods and number of variables
    for select_method in list_methods:
        for n_var in list_n:
            print("=" * 110)
            print(f"HORIZON = {horizon}")
            print(f"METHOD = {select_method}")
            print(f"NUMBER OF VARIABLES = {n_var}")
            print("=" * 110)
            
            # Determine indices for out-of-sample exercise using the provided dates
            try:
                n_start = data_real.index[data_real['date'] == pd.to_datetime(start_date_oos)][0]
                n_end = data_real.index[data_real['date'] == pd.to_datetime(end_date_oos)][0]
            except IndexError:
                raise ValueError("Start or end date for out-of-sample not found in the dataset.")
            
            # Create a results DataFrame to store out-of-sample predictions.
            # (We initially create placeholder columns; these will be updated once the first regression run returns column names.)
            num_rows = n_end - n_start + 1
            results = pd.DataFrame(np.nan, index=range(num_rows), columns=["date"])
            
            # Loop over out-of-sample dates (each iteration corresponds to one prediction date)
            for idx, ii in enumerate(range(n_start, n_end + 1)):
                # Use all available data up to the current time point
                data_all = data_real.iloc[:ii]
                date_ii = data_all['date'].iloc[-1]
                year = date_ii.year
                month = date_ii.month
                print(f"Doing out-of-sample predictions for {year} at month {month}")
                
                # --------------------------------------------------------------------
                # STEP A: Pre-selection
                # --------------------------------------------------------------------
                var_sel = pre_select(data_real, ii, horizon, select_method, n_var)
                if len(var_sel) != n_var and select_method != 2:
                    raise Exception("Pre-selection step did not work, please check")
                
                # Split the data into regressors (RHS) and target (LHS)
                rhs_sel = data_all[['date'] + var_sel]
                lhs_sel = data_all[['date', 'target', 'L1st_target', 'L2nd_target']]
                
                # --------------------------------------------------------------------
                # STEP B: Factor extraction (PCA) on pre-selected variables
                # --------------------------------------------------------------------
                # Remove any rows with missing values in the regressor set
                x_pca = rhs_sel.dropna()
                start_date_val = x_pca['date'].iloc[0]
                end_date_val = x_pca['date'].iloc[-1]
                # Restrict the LHS data to the same date range as the PCA data
                lhs_sel = lhs_sel[(lhs_sel['date'] >= start_date_val) & (lhs_sel['date'] <= end_date_val)]
                
                if do_factors == 1:
                    # Drop the date column before running PCA
                    x_pca_nodate = x_pca.drop(columns=['date'])
                    scaler = StandardScaler()
                    x_scaled = scaler.fit_transform(x_pca_nodate)
                    # Run PCA on all available components
                    pca = PCA(n_components=x_pca_nodate.shape[1])
                    pca_result = pca.fit_transform(x_scaled)
                    # Create a DataFrame for PCA results with component names
                    pca_cols = [f"PC{i+1}" for i in range(x_pca_nodate.shape[1])]
                    rhs_fct = pd.DataFrame(pca_result, index=x_pca.index, columns=pca_cols)
                    # Keep only factors with eigenvalue > 1 (Kaiser criterion)
                    valid_factors = [col for col, eigen in zip(pca_cols, pca.explained_variance_) if eigen > 1]
                    rhs_fct_sel = rhs_fct[valid_factors]
                    # Combine LHS and factor components into the final dataset for regression
                    don_cb = pd.concat([lhs_sel.reset_index(drop=True), rhs_fct_sel.reset_index(drop=True)], axis=1)
                else:
                    # If not extracting factors, simply use the available regressors (excluding the date)
                    don_cb = pd.concat([lhs_sel.reset_index(drop=True), x_pca.drop(columns=['date']).reset_index(drop=True)], axis=1)
                
                # --------------------------------------------------------------------
                # STEP C: Regression on factors
                # --------------------------------------------------------------------
                don_reg = don_cb.dropna()
                # Identify the row corresponding to the current date
                n_date_idx = don_reg.index[don_reg['date'] == date_ii]
                if len(n_date_idx) == 0:
                    continue
                n_date_idx = n_date_idx[0]
                
                # Define the in-sample and out-of-sample datasets.
                # (Note: In the original R code, "head(don_reg, n_date_idx - horizon - 3)" is used.)
                smpl_in = don_reg.iloc[: max(n_date_idx - horizon - 3, 0)]
                smpl_out = don_reg.iloc[n_date_idx]
                results.at[idx, 'date'] = date_ii
                
                # Run the regressions.
                # The run_regressions function is assumed to return a dict or Series of forecast results.
                temp_res = run_regressions(smpl_in, smpl_out, list_reg, n_var, sc_ml, fast_MRF)
                if isinstance(temp_res, dict):
                    temp_res = pd.Series(temp_res)
                # On the first prediction, update the results DataFrame with proper column names
                if idx == 0:
                    new_cols = ['date'] + list(temp_res.index)
                    results = results.reindex(columns=new_cols)
                results.loc[idx, temp_res.index] = temp_res.values
            
            # After looping over all out-of-sample dates, compute the mean prediction (if applicable)
            pred_cols = [col for col in results.columns if col.startswith("pred_")]
            if pred_cols:
                results['pred_mean'] = results[pred_cols].mean(axis=1)
            results['date'] = pd.to_datetime(results['date'])
            results['year'] = results['date'].dt.year
            
            # ----------------------------------------------------------------------------
            # Write predictions and RMSE outputs
            # ----------------------------------------------------------------------------
            output_dir = os.path.join("2-Output", f"h{horizon}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Write out-of-sample predictions to CSV
            pred_filename = (
                f"pred_sel_{select_method}_n_{n_var}_reg_{'_'.join(map(str, list_reg))}_"
                f"h_{horizon}_{start_date_oos}_{end_date_oos}.csv"
            )
            results.drop(columns=['year']).to_csv(os.path.join(output_dir, pred_filename), index=False)
            
            # Define RMSE function
            def err(X, Y):
                return np.sqrt(np.mean((X - Y) ** 2))
            
            # Assuming the second column in results corresponds to the true target values
            true_vals = results.iloc[:, 1]
            rmse_total = {col: err(results[col], true_vals) for col in results.columns if col not in ['date', 'year']}
            
            crisis_years = [2008, 2009, 2020, 2021]
            crisis_data = results[results['year'].isin(crisis_years)]
            rmse_crisis = {col: err(crisis_data[col], crisis_data.iloc[:, 1]) for col in crisis_data.columns if col not in ['date', 'year']}
            
            normal_data = results[~results['year'].isin(crisis_years)]
            rmse_normal = {col: err(normal_data[col], normal_data.iloc[:, 1]) for col in normal_data.columns if col not in ['date', 'year']}
            
            # Combine RMSE summaries into a DataFrame and remove the 'true_value' entry if present
            summary_all = pd.DataFrame({
                'total': rmse_total,
                'crisis': rmse_crisis,
                'normal': rmse_normal
            })
            if 'true_value' in summary_all.index:
                summary_all = summary_all.drop(index='true_value')
            
            # Write RMSE summary to CSV
            rmse_filename = (
                f"rmse_sel_{select_method}_n_{n_var}_reg_{'_'.join(map(str, list_reg))}_"
                f"h_{horizon}_{start_date_oos}_{end_date_oos}.csv"
            )
            summary_all.to_csv(os.path.join(output_dir, rmse_filename))
            
            # Record results in a summary dictionary.
            # (Here, we mimic the R summary matrix: first row pre-selection method, second row number of variables,
            #  subsequent rows the RMSE values for the total sample.)
            key = f"sel_{select_method}_n_{n_var}"
            summary_ps_meth[key] = {
                'pre_selection': select_method,
                'nb_variables': n_var,
                'rmse_total': summary_all['total'].to_dict()
            }
        # End loop for n_var
    # End loop for pre-selection method
    
    # Write an overall summary for this horizon
    summary_df = pd.DataFrame(summary_ps_meth).T.reset_index(drop=True)
    summary_filename = (
        f"summaryALL_sel_{'_'.join(map(str, list_methods))}_n_{'_'.join(map(str, list_n))}_"
        f"reg_{'_'.join(map(str, list_reg))}_h_{horizon}_{start_date_oos}_{end_date_oos}.csv"
    )
    summary_df.to_csv(os.path.join("2-Output", f"h{horizon}", summary_filename), index=False)
    
    # Save the horizon summary for further use if needed
    summary_ps_meth_all[f"h{horizon}"] = summary_df

# End of processing
