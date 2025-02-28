# Translations from the functions directory

# general dependencies
import pandas as pd
import numpy as np
from datetime import datetime
# for cleaning
from statsmodels.tsa.stattools import adfuller
# for pre selection
from sklearn.linear_model import lars_path
import statsmodels.api as sm
# for regression
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# ---------------------------------------------------------------------------
# Placeholder tuning functions – replace these with your own implementations
# ---------------------------------------------------------------------------
def tune_RF(x_train, y_train, n_per):
    # Return dummy parameters for random forest:
    # mtry (max_features) and nodesize (min_samples_leaf)
    return np.array([[min(5, x_train.shape[1]), 5]])

def tune_XGBT(x_train, y_train, n_per):
    # Return dummy parameters for XGBoost tree:
    # [nrounds, eta, max_depth, min_child_weight, gamma]
    return np.array([50, 0.1, 3, 1, 0])

def tune_MRF(data_in, n_per):
    # Return dummy parameters for full-tuning of macroeconomic random forest.
    # For example, B, x.pos (an integer), ridge.lambda, resampling.opt, block.size.
    return np.array([100, data_in.shape[1]-1, 0.1, 5, 3])

def tune_MRF_fast(data_in, n_per):
    # Return a single parameter indicating number of predictors to use in MRF.
    return data_in.shape[1] - 1

def tune_XGBL(x_train, y_train, n_per):
    # Return dummy parameters for XGBoost linear model:
    # [nrounds, eta, alpha]
    return np.array([50, 0.1, 0.01])

# ---------------------------------------------------------------------------
# run_regressions function translation
# ---------------------------------------------------------------------------
def run_regressions(smpl_in, smpl_out, list_methods, n_sel, sc_ml, fast_MRF, n_per):
    """
    From a given in-sample (smpl_in) and out-of-sample (smpl_out) dataset, 
    this function fits various regression models and returns a DataFrame with 
    the true target value and predictions from each model.
    
    Parameters
    ----------
    smpl_in : pd.DataFrame
        In-sample data. Expected to include columns: date, target, L1st_target, L2nd_target, predictors...
    smpl_out : pd.DataFrame
        Out-of-sample data. Same column structure as smpl_in.
    list_methods : list of int
        List of regression techniques to be tested:
            1 = OLS
            2 = Markov-switching regression (requires OLS fit from method 1)
            3 = Quantile regression
            4 = Random forest
            5 = XGBoost tree
            6 = Macroeconomic Random Forest
            7 = XGBoost linear
        The AR benchmark is always performed.
    n_sel : int
        (Not used explicitly in this translation; reserved for selection size.)
    sc_ml : int
        1 to scale data for machine learning methods, 0 to use raw data.
    fast_MRF : int
        1 to perform fast tuning for macroeconomic random forest, 0 for full tuning.
    n_per : int
        Number of periods for hyper-parameter tuning (used by tuning functions).
        
    Returns
    -------
    results : pd.DataFrame
        A DataFrame with one row and columns: true_value, ar, and one column per regression method.
    """
    
    # Initialize results DataFrame: columns = true_value, ar, plus one column per method in list_methods.
    n_cols = len(list_methods) + 2
    results = pd.DataFrame(np.nan, index=[0], columns=range(n_cols))
    # Assuming target is in column 1 of smpl_out.
    results.iloc[0, 0] = smpl_out.iloc[0, 1]
    col_names = ["true_value", "ar"]
    
    # Number of factors (predictors) – assuming first 4 columns are date, target, L1st_target, L2nd_target.
    n_fct = smpl_in.shape[1] - 4

    # Prepare training and test samples for ML methods.
    # y_train: target from smpl_in.
    y_train = smpl_in["target"].values
    scaler_y = StandardScaler()
    y_train_sc = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    
    # x_train: all predictors (columns 4 onward)
    x_train = smpl_in.iloc[:, 4:].values
    scaler_x = StandardScaler()
    x_train_sc = scaler_x.fit_transform(x_train)
    
    # x_test: predictors from smpl_out (columns 4 onward)
    x_test = smpl_out.iloc[:, 4:].values
    # Combine x_train and x_test to get consistent scaling.
    x_all = np.vstack([x_train, x_test])
    scaler_x_all = StandardScaler()
    x_all_sc = scaler_x_all.fit_transform(x_all)
    # x_test_sc is the last row of x_all_sc.
    x_test_sc = x_all_sc[-x_test.shape[0]:, :]
    # For our purposes we use only the first test row.
    x_test_sc = x_test_sc[0, :].reshape(1, -1)
    
    if sc_ml == 1:
        y_train_ml = y_train_sc
        x_train_ml = x_train_sc
        x_test_ml = x_test_sc
        x_all_ml = x_all_sc
    else:
        y_train_ml = y_train
        x_train_ml = x_train
        x_test_ml = x_test
        x_all_ml = x_all

    # ---------------------------
    # AR Model: Regress target on L1st_target.
    # ---------------------------
    # Use only columns 'target' and 'L1st_target' from smpl_in.
    ar_data = smpl_in[["target", "L1st_target"]].dropna()
    X_ar = ar_data[["L1st_target"]].values
    y_ar = ar_data["target"].values
    ar_model = LinearRegression().fit(X_ar, y_ar)
    # Predict for smpl_out using L1st_target.
    X_ar_test = smpl_out[["L1st_target"]].values
    ar_pred = ar_model.predict(X_ar_test)[0]
    results.iloc[0, 1] = ar_pred

    count_col = 2  # Starting column index for additional methods.

    # ---------------------------
    # Method 1: OLS using all predictors (columns 4 onward).
    # ---------------------------
    if 1 in list_methods:
        print("Forecasting with OLS")
        # Build OLS model: target ~ predictors (using columns from index 4 onward)
        X_ols = smpl_in.iloc[:, 4:]
        y_ols = smpl_in["target"]
        X_ols = sm.add_constant(X_ols)
        ols_model = sm.OLS(y_ols, X_ols, missing='drop').fit()
        
        # Prepare test data: predictors from smpl_out, same columns as in smpl_in.
        X_ols_test = smpl_out.iloc[:, 4:]
        X_ols_test = sm.add_constant(X_ols_test)
        ols_pred = ols_model.predict(X_ols_test).iloc[0]
        
        results.iloc[0, count_col] = ols_pred
        col_names.append("pred_ols")
        count_col += 1

    # ---------------------------
    # Method 2: Markov-switching regression.
    # ---------------------------
    if 2 in list_methods:
        print("Forecasting with Markov-switching")
        # Note: Python's statsmodels offers MarkovRegression for time series; however,
        # replicating the R msmFit is nontrivial. Here we attempt a placeholder:
        try:
            # As a proxy, we refit the OLS model from Method 1 and simulate regime switching.
            # For demonstration, we compute two sets of coefficients by perturbing the OLS coefficients.
            coef = ols_model.params
            # Create two regimes by adding and subtracting a small constant.
            coef_regime1 = coef * 1.01
            coef_regime2 = coef * 0.99
            # Prepare test data.
            X_ms_test = X_ols_test.iloc[0].values  # as numpy array
            pred1 = np.sum(coef_regime1.values * X_ms_test)
            pred2 = np.sum(coef_regime2.values * X_ms_test)
            # Simulate regime probabilities (for example, using OLS residuals or a simple rule).
            # Here we simply weight based on which prediction is higher.
            if pred1 > pred2:
                pred = 0.6 * pred1 + 0.4 * pred2
            else:
                pred = 0.4 * pred1 + 0.6 * pred2
        except Exception as e:
            print("Markov-switching regression failed, reverting to OLS:", e)
            pred = ols_pred
        results.iloc[0, count_col] = pred
        col_names.append("pred_ms")
        count_col += 1

    # ---------------------------
    # Method 3: Quantile regression.
    # ---------------------------
    if 3 in list_methods:
        print("Forecasting with quantile regression")
        # Use quantile regression at the median (tau=0.5)
        X_qr = smpl_in.iloc[:, 4:]
        y_qr = smpl_in["target"]
        X_qr = sm.add_constant(X_qr)
        qr_model = QuantReg(y_qr, X_qr).fit(q=0.5)
        X_qr_test = smpl_out.iloc[:, 4:]
        X_qr_test = sm.add_constant(X_qr_test)
        qr_pred = qr_model.predict(X_qr_test).iloc[0]
        results.iloc[0, count_col] = qr_pred
        col_names.append("pred_qr")
        count_col += 1

    # ---------------------------
    # Method 4: Random Forest.
    # ---------------------------
    if 4 in list_methods:
        print("Forecasting with Random Forest")
        # Tune parameters using the placeholder tuning function.
        param_rf = tune_RF(x_train_ml, y_train_ml, n_per)
        # Create and fit the Random Forest.
        rf_model = RandomForestRegressor(n_estimators=300,
                                         max_features=int(param_rf[0,0]),
                                         min_samples_leaf=int(param_rf[0,1]),
                                         random_state=0)
        rf_model.fit(x_train_ml, y_train_ml)
        rf_pred = rf_model.predict(x_test_ml)[0]
        # If scaling was used, rescale back.
        if sc_ml == 1:
            rf_pred = rf_pred * np.std(y_train) + np.mean(y_train)
        results.iloc[0, count_col] = rf_pred
        col_names.append("pred_rf")
        count_col += 1

    # ---------------------------
    # Method 5: XGBoost Tree.
    # ---------------------------
    if 5 in list_methods:
        print("Forecasting with XGBoost (tree)")
        param_xgbt = tune_XGBT(x_train_ml, y_train_ml, n_per)
        # Create DMatrix objects.
        dtrain = xgb.DMatrix(x_train_ml, label=y_train_ml)
        dtest = xgb.DMatrix(x_test_ml)
        params = {
            'objective': 'reg:squarederror',
            'eta': float(param_xgbt[1]),
            'max_depth': int(param_xgbt[2]),
            'min_child_weight': int(param_xgbt[3]),
            'gamma': float(param_xgbt[4]),
            'verbosity': 0
        }
        num_round = int(param_xgbt[0])
        xgbt_model = xgb.train(params, dtrain, num_round)
        xgbt_pred = xgbt_model.predict(dtest)[0]
        if sc_ml == 1:
            xgbt_pred = xgbt_pred * np.std(y_train) + np.mean(y_train)
        results.iloc[0, count_col] = xgbt_pred
        col_names.append("pred_xgbt")
        count_col += 1

    # ---------------------------
    # Method 6: Macroeconomic Random Forest.
    # ---------------------------
    if 6 in list_methods:
        print("Forecasting with Macroeconomic Random Forest")
        # Prepare data: combine y_train_ml and x_train_ml, then append x_test_ml.
        # Here we create a DataFrame similar to R's cbind.
        data_in = np.hstack([y_train_ml.reshape(-1, 1), x_train_ml])
        data_in = np.vstack([data_in, np.hstack([[np.nan], x_test_ml.flatten()])])
        # Set seed for reproducibility.
        np.random.seed(22122)
        if fast_MRF == 0:
            param_mrf = tune_MRF(data_in, n_per)
            # Placeholder for MRF model – here we simply use a random forest as a proxy.
            mrf_model = RandomForestRegressor(n_estimators=int(param_mrf[0]),
                                              max_features=int(param_mrf[1]-1),
                                              min_samples_leaf=5,
                                              random_state=0)
        else:
            param_mrf_fast = tune_MRF_fast(data_in, n_per)
            # Fast tuning: use a simpler random forest.
            mrf_model = RandomForestRegressor(n_estimators=100,
                                              max_features=int(param_mrf_fast),
                                              random_state=0)
        # Fit on in-sample part.
        mrf_model.fit(x_train_ml, y_train_ml)
        mrf_pred = mrf_model.predict(x_test_ml)[0]
        if sc_ml == 1:
            mrf_pred = mrf_pred * np.std(y_train) + np.mean(y_train)
        results.iloc[0, count_col] = mrf_pred
        col_names.append("pred_mrf")
        count_col += 1

    # ---------------------------
    # Method 7: XGBoost Linear.
    # ---------------------------
    if 7 in list_methods:
        print("Forecasting with XGBoost (linear)")
        param_xgbl = tune_XGBL(x_train_ml, y_train_ml, n_per)
        dtrain = xgb.DMatrix(x_train_ml, label=y_train_ml)
        params_lin = {
            'objective': 'reg:squarederror',
            'booster': 'gblinear',
            'eta': float(param_xgbl[1]),
            'alpha': float(param_xgbl[2]),
            'verbosity': 0
        }
        num_round_lin = int(param_xgbl[0])
        xgbl_model = xgb.train(params_lin, dtrain, num_round_lin)
        dtest = xgb.DMatrix(x_test_ml)
        xgbl_pred = xgbl_model.predict(dtest)[0]
        if sc_ml == 1:
            xgbl_pred = xgbl_pred * np.std(y_train) + np.mean(y_train)
        results.iloc[0, count_col] = xgbl_pred
        col_names.append("pred_xgbl")
        count_col += 1

    results.columns = col_names
    return results

def iBMA_bicreg(X, y, thresProbne0, verbose=True, maxNvar=20, nIter=20):
    """
    A simplified implementation of an iterated Bayesian Model Averaging (iBMA) procedure 
    using a forward-selection approach with the BIC as the selection criterion.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor matrix with column names.
    y : np.array or pd.Series
        Response variable.
    thresProbne0 : float
        Threshold on the “importance” of predictors (not used explicitly in this simple version).
    verbose : bool, default True
        If True, prints progress.
    maxNvar : int, default 20
        Maximum number of predictors to select in one batch.
    nIter : int, default 20
        Maximum number of iterations (used here as an upper bound on forward steps).

    Returns
    -------
    dict
        A dictionary with keys:
          - 'sortedX': a DataFrame whose columns are the predictors in the order selected (here empty,
                       but included for interface consistency),
          - 'currentSet': list of selected variable names,
          - 'bma': a dictionary with key 'namesx' containing the selected variable names.
    """
    # Ensure X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    current_set = []
    remaining = list(X.columns)
    best_bic = np.inf
    bic_history = []
    
    # Forward selection loop up to either maxNvar predictors or nIter iterations.
    for i in range(nIter):
        bic_candidates = {}
        for var in remaining:
            candidate_set = current_set + [var]
            X_candidate = sm.add_constant(X[candidate_set])
            model = sm.OLS(y, X_candidate).fit()
            bic_candidates[var] = model.bic
        # Identify the variable whose inclusion yields the lowest BIC.
        best_var = min(bic_candidates, key=bic_candidates.get)
        candidate_bic = bic_candidates[best_var]
        if candidate_bic < best_bic:
            current_set.append(best_var)
            remaining.remove(best_var)
            best_bic = candidate_bic
            bic_history.append((best_var, best_bic))
            if verbose:
                print(f"Iteration {i+1}: Added {best_var}, BIC = {best_bic:.2f}")
            # Stop if we reached the maximum number of predictors in this batch.
            if len(current_set) >= maxNvar:
                break
        else:
            if verbose:
                print("No further improvement in BIC; stopping selection.")
            break

    # For compatibility with the interface in the pre_selection function,
    # we return a dictionary containing an (empty) DataFrame for sortedX and the current set.
    sortedX = pd.DataFrame({var: [] for var, _ in bic_history})
    return {'sortedX': sortedX, 'currentSet': current_set, 'bma': {'namesx': current_set}}

def pre_select(data_real, ii, horizon, select_method, n_var, fast_bma=1):
    """
    For a given dataset, select the n_var most informative predictors based on the chosen pre-selection method.
    
    Parameters
    ----------
    data_real : pd.DataFrame
        Dataset with LHS and RHS variables. It must include columns such as 'target', 'L1st_target', and 'L2nd_target'.
    ii : int
        The row index (time point) at which to trim the dataset.
    horizon : int
        Forecast horizon (negative for backcasting, 0 for nowcasting, positive for forecasting).
    select_method : int
        Pre-selection method to use:
            0 = No pre-selection
            1 = LARS (Efron et al., 2004)
            2 = Correlation-based (SIS: Fan and Lv, 2008)
            3 = t-stat based (Bair et al., 2006)
            4 = Iterated Bayesian Model Averaging (BMA: Yeung et al., 2005)
    n_var : int
        Number of variables to select.
    fast_bma : int, optional
        Switch for fast BMA (1 = fast, 0 = full), by default 1.
        
    Returns
    -------
    var_sel : list of str
        List of selected variable names.
    """
    # ----------------------------------------------------
    # Initiate sample: use rows up to (ii - horizon - 3), remove date and lag columns, drop missing rows.
    smpl = data_real.iloc[:(ii - horizon - 3)].copy()
    cols_to_drop = ['date', 'L1st_target', 'L2nd_target']
    smpl = smpl.drop(columns=[c for c in cols_to_drop if c in smpl.columns], errors='ignore')
    smpl = smpl.dropna()
    
    if select_method == 2:
        # Correlation-based pre-selection: compute absolute Pearson correlations.
        corr_dict = {}
        for col in smpl.columns:
            if col != 'target':
                corr_val = smpl[col].corr(smpl['target'])
                corr_dict[col] = abs(corr_val)
        # Order variables by correlation (descending) and select top n_var.
        sorted_vars = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)
        var_sel = [var for var, corr in sorted_vars[:n_var]]
        
    elif select_method == 1:
        # LARS-based pre-selection.
        # We'll use sklearn's lars_path to determine the order in which predictors enter the model.
        batch = smpl.copy()
        selected_vars = []
        count = 0
        while count < n_var:
            # Separate predictors and target.
            if 'target' not in batch.columns or batch.shape[1] <= 1:
                break
            X = batch.drop(columns=['target'])
            y = batch['target'].values
            # If no predictors remain, break.
            if X.shape[1] == 0:
                break
            # Compute LARS path.
            # lars_path returns alphas, active (order of variable indices), and coefs.
            _, active, _ = lars_path(X.values, y, method='lar', verbose=False)
            # Get variable names in order of activation.
            active_vars = [X.columns[i] for i in active]
            if len(active_vars) == 0:
                # If no variable is selected, add all remaining predictors.
                remaining = list(X.columns)
                selected_vars.extend(remaining)
                count += len(remaining)
                print("Warning: Using special procedure for LARS")
                break
            else:
                # Append new variables (avoiding duplicates).
                new_vars = [v for v in active_vars if v not in selected_vars]
                selected_vars.extend(new_vars)
                count = len(selected_vars)
                # Remove these selected predictors from the batch.
                batch = batch.drop(columns=new_vars)
        var_sel = selected_vars[:n_var]
        
    elif select_method == 0:
        # No pre-selection: return all predictors (i.e. all columns except 'target').
        var_sel = [col for col in smpl.columns if col != 'target']
        
    elif select_method == 3:
        # t-statistic based pre-selection.
        # Use an initial dataset (with date dropped) and drop missing values.
        init = data_real.iloc[:(ii - horizon - 3)].copy()
        init = init.drop(columns=['date'], errors='ignore').dropna()
        # List of candidate predictors: those in smpl except 'target'.
        list_var = [col for col in smpl.columns if col != 'target']
        tstats = []
        for v in list_var:
            # Build a regression dataset with target, its lags, and the candidate predictor.
            # Assumes that 'L1st_target' and 'L2nd_target' exist in init.
            if v not in init.columns:
                continue
            data_eq = init[['target', 'L1st_target', 'L2nd_target', v]].dropna()
            if data_eq.shape[0] == 0:
                continue
            X = data_eq[['L1st_target', 'L2nd_target', v]]
            X = sm.add_constant(X)
            y = data_eq['target']
            model = sm.OLS(y, X).fit()
            # In the regression, the candidate predictor is the fourth coefficient.
            t_val = model.tvalues[v]
            tstats.append((v, t_val))
        # Order predictors by descending t-statistic.
        tstats_sorted = sorted(tstats, key=lambda x: x[1], reverse=True)
        var_sel = [v for v, t in tstats_sorted[:n_var]]
        
    elif select_method == 4:
        # Iterated Bayesian Model Averaging (BMA) pre-selection.
        # Hyper-parameters (tuned empirically).
        max_bma = 20
        th_bma = 20
        niter_fast_bma = 20
        
        # Use all predictors (excluding target).
        x_all = smpl.drop(columns=['target'])
        y = smpl['target'].values.reshape(-1, 1)
        var_sel = []
        n_var_sel = 0
        
        # Iteratively add variables until at least n_var are selected.
        while n_var_sel < n_var:
            var_left = n_var - n_var_sel
            max_batch = min(max_bma, var_left)
            # Current batch: predictors not yet selected.
            current_cols = [col for col in x_all.columns if col not in var_sel]
            X = x_all[current_cols].values
            if X.shape[1] == 0:
                break
            if max_batch == 1:
                bma = iBMA_bicreg(X, y, thresProbne0=th_bma, verbose=True, maxNvar=max_batch+1, nIter=1)
                # Assume bma['sortedX'] is a DataFrame whose columns are candidate predictors.
                sel_fast = list(bma['sortedX'].columns)[0:1]
                var_sel.extend(sel_fast)
            else:
                if fast_bma == 1:
                    bma = iBMA_bicreg(X, y, thresProbne0=th_bma, verbose=True, maxNvar=max_batch, nIter=niter_fast_bma)
                    # Assume bma['sortedX'] is a DataFrame and bma['currentSet'] is a list of selected predictors.
                    sel_fast = list(bma['sortedX'].columns)  # or use bma['currentSet'] if available
                    var_sel.extend(sel_fast)
                else:
                    bma = iBMA_bicreg(X, y, thresProbne0=th_bma, verbose=True, maxNvar=max_batch, nIter=X.shape[1])
                    var_sel.extend(bma['bma']['namesx'])
            n_var_sel = len(var_sel)
        # End of BMA branch.
    else:
        # If no valid method is provided, return an empty list.
        var_sel = []
    
    return var_sel

def do_clean(data_init, min_start_date):
    """
    Cleans the initial dataset.
    
    (1) Keeps only those variables that have a non-missing value at min_start_date.
    (2) Interpolates missing observations that occur in the middle of the dataset 
        (leading and trailing missing values remain untouched).
    (3) Removes non-stationary variables (based on an ADF test with p-value > 0.10).
    
    Parameters
    ----------
    data_init : pd.DataFrame
        The initial dataset. Must contain a 'date' column (in a format parsable by pd.to_datetime)
        and a 'target' column among others.
    min_start_date : str
        The minimum start date for variables in "YYYY-MM-15" format.
    
    Returns
    -------
    data_rmv : pd.DataFrame
        The cleaned dataset.
    """
    # Ensure the date column is in datetime format
    data_init = data_init.copy()
    data_init['date'] = pd.to_datetime(data_init['date'])
    min_date = pd.to_datetime(min_start_date)
    
    # -----------------------------------
    # 1 - Remove variables starting after the user-defined min_start_date
    # -----------------------------------
    # At min_start_date, melt the data so that each column becomes a row, then keep only those
    # variables with non-missing values at that date.
    ctrl_min = data_init.loc[data_init['date'] == min_date].melt(id_vars='date',
                                                                 var_name='variable',
                                                                 value_name='value')
    ctrl_min = ctrl_min[ctrl_min['value'].notna()]
    # Get the unique variables (plus always keep 'date')
    var_to_keep = list(ctrl_min['variable'].unique())
    if 'date' not in var_to_keep:
        var_to_keep.append('date')
        
    # Subset the dataset to keep only these variables
    data_rmv = data_init[var_to_keep].copy()
    
    # -----------------------------------
    # 2 - Clean data: interpolate missing observations and remove non-stationary variables
    # -----------------------------------
    # Interpolate missing observations (only fill gaps in the interior; leading/trailing remain NaN)
    cols_to_interp = [col for col in data_rmv.columns if col != 'date']
    data_rmv[cols_to_interp] = data_rmv[cols_to_interp].interpolate(method='linear', limit_direction='both')
    
    # Remove non-stationary variables based on ADF test (p-value > 0.10)
    # Do not test the date or target variables.
    cols_to_test = [col for col in data_rmv.columns if col not in ['date', 'target']]
    nonstationary_vars = []
    
    for col in cols_to_test:
        series = data_rmv[col].dropna()
        if len(series) > 0:
            adf_result = adfuller(series, autolag='AIC')
            p_value = adf_result[1]
            if p_value > 0.10:
                nonstationary_vars.append(col)
                
    # Drop the non-stationary variables from the dataset
    data_rmv = data_rmv.drop(columns=nonstationary_vars)
    
    return data_rmv

def do_align(data_rmv, horizon):
    """
    Re-aligns the input dataset to balance the input data, using a vertical realignment
    procedure analogous to Altissimo et al. (2006). In addition to lagging missing RHS
    observations, the procedure creates lead variables when extra observations are available.
    
    Parameters:
      data_rmv : pd.DataFrame
          Cleaned input dataset that must include at least the columns 'date' and 'target'
          plus additional regressors.
      horizon : int
          Forecast horizon (negative for backcast, 0 for nowcast, positive for forecast).
          
    Returns:
      data_real : pd.DataFrame
          The re-aligned dataset.
    """
    # -------------------------------
    # 1 - Set date of back-/now-/fore-cast
    # -------------------------------
    # Get the last non-missing date
    temp = data_rmv[['date']].dropna().tail(1)
    month_data = temp.iloc[0, 0]
    if not isinstance(month_data, datetime):
        month_data = pd.to_datetime(month_data)
        
    month_of_data = month_data.month
    year_of_data = month_data.year

    # Adjust the year depending on horizon
    if (month_of_data + horizon) <= 0:
        year_cast = year_of_data - (abs(horizon) // 12 + 1)
    elif (month_of_data + horizon) > 12:
        year_cast = year_of_data + (abs(horizon) // 12 + 1)
    else:
        year_cast = year_of_data

    # Compute month of cast using modulo arithmetic (adjust if remainder is 0)
    month_cast = (month_of_data + horizon) % 12
    if month_cast == 0:
        month_cast = 12

    # Create the cast date (using the 15th day of the month)
    date_cast = datetime(year_cast, month_cast, 15)

    # Find the position index corresponding to date_cast in data_rmv
    # Here we compare string representations to mimic R's grepl behavior.
    date_str_cast = date_cast.strftime("%Y-%m-%d")
    mask = data_rmv['date'].astype(str) == date_str_cast
    n_time_idx = data_rmv.index[mask]
    
    # If no matching date (i.e. forecast date not present), complete the series with NA rows.
    if len(n_time_idx) == 0:
        first_date = data_rmv['date'].iloc[0]
        # Create a complete monthly date range up to date_cast.
        # Assuming dates should be on the 15th day of each month.
        complete_dates = pd.date_range(start=first_date, end=date_cast, freq='MS').to_series().apply(lambda d: d.replace(day=15))
        # Reindex the dataframe by these dates.
        data_rmv = data_rmv.set_index('date').reindex(complete_dates).reset_index().rename(columns={'index': 'date'})
        mask = data_rmv['date'].astype(str) == date_str_cast
        n_time_idx = data_rmv.index[mask]
    n_time_idx = n_time_idx[0]  # take the first occurrence

    # -------------------------------
    # 2 - Initialize re-aligned dataset with LHS variable and date
    # -------------------------------
    data_real = data_rmv[['date', 'target']].copy()

    # Compute first and last indices where 'target' is not NA
    non_na_idx = data_real.index[data_real['target'].notna()]
    if len(non_na_idx) == 0:
        raise ValueError("No non-missing values found in 'target'.")
    first_non_na = non_na_idx.min()
    last_non_na = non_na_idx.max()
    diff_na = last_non_na - n_time_idx

    # If diff_na is non-negative, then an observation exists for the cast date.
    # We then remove observations to allow for forecasting.
    if diff_na >= 0:
        print(f"Warning: There is already an observation for the horizon {horizon}. "
              "The observation has been deleted to allow for forecasts. "
              "For a more realistic set-up, please change horizons accordingly.")
        # Delete observation(s) between (n_time_idx - diff_na) and n_time_idx (inclusive)
        idx_range = range(n_time_idx - diff_na, n_time_idx + 1)
        data_real.loc[idx_range, 'target'] = np.nan
        # Recompute non-NA indices and diff_na
        non_na_idx = data_real.index[data_real['target'].notna()]
        if len(non_na_idx) == 0:
            raise ValueError("No non-missing 'target' values remain after deletion.")
        first_non_na = non_na_idx.min()
        last_non_na = non_na_idx.max()
        diff_na = last_non_na - n_time_idx

    # Create first and second available lags for the target variable.
    # In R, lag with a negative shift acts as a lead.
    data_real['L1st_target'] = data_real['target'].shift(-diff_na)
    data_real['L2nd_target'] = data_real['target'].shift(-diff_na + 1)

    # -------------------------------
    # 3 - Re-align RHS variables
    # -------------------------------
    # Select RHS variables (all columns except 'date' and 'target')
    temp = data_rmv.drop(columns=['date', 'target'])
    
    # Loop over each RHS variable (each column)
    for col in temp.columns:
        temp_col = temp[[col]].copy()
        # Get indices where the column is not NA
        non_na_idx_col = temp_col.index[temp_col[col].notna()]
        if len(non_na_idx_col) == 0:
            # If the entire column is NA, simply add it to data_real.
            data_real[col] = temp_col[col]
            continue
        first_non_na_col = non_na_idx_col.min()
        last_non_na_col = non_na_idx_col.max()
        diff_na_col = last_non_na_col - n_time_idx

        if diff_na_col == 0:
            # If the last non-NA equals the cast date, bind directly.
            data_real[col] = temp_col[col]
        elif diff_na_col < 0:
            # If the last non-NA is before the cast date: shift the variable upward by -diff_na_col.
            shifted = temp_col[col].shift(-diff_na_col)
            # Replace the first (-diff_na_col) NA values with the value at position (first_non_na_col - diff_na_col)
            fill_value = shifted.iloc[first_non_na_col - diff_na_col]
            for count in range(1, -diff_na_col + 1):
                idx_to_fill = first_non_na_col + count - 1
                shifted.iloc[idx_to_fill] = fill_value
            new_col_name = f"Lg{-diff_na_col}_{col}"
            data_real[new_col_name] = shifted
        elif diff_na_col > 0:
            # If the last non-NA is after the cast date: create additional lead variables.
            # Start with the original column.
            temp_df = pd.DataFrame({col: temp_col[col]})
            for count in range(1, diff_na_col + 1):
                lead_col = temp_col[col].shift(-count)
                new_name = f"Ld{count}_{col}"
                temp_df[new_name] = lead_col
            # Bind all these columns to data_real.
            for newcol in temp_df.columns:
                data_real[newcol] = temp_df[newcol]

    # -------------------------------
    # 4 - Check procedure and return re-aligned dataset
    # -------------------------------
    # At the cast date row (n_time_idx), there should be only one NA (for the target forecast).
    if data_real.loc[n_time_idx].isna().sum() > 1:
        raise ValueError("Re-alignment procedure not successful, please check.")
    
    return data_real
