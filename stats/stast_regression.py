
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from typing import List, Dict, Tuple, Any
warnings.filterwarnings('ignore')

DB_CONFIG = {
    'host': 'localhost',
    'database': 'dcb',
    'user': 'doriangarin',
    'password': '96349130dG!',
    'port': 5432
}
# For standalone execution, include these functions:
import os
from sqlalchemy import create_engine


def create_directories():
    """Create directories for saving results and plots"""
    os.makedirs('stats/plots', exist_ok=True)
    os.makedirs('stats/results', exist_ok=True)
    os.makedirs('stats/models', exist_ok=True)

def connect_to_db():
    """Create database connection"""
    try:
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def load_data(engine):
    """Load data from camilla2 and denovo tables with inner join on DOB"""
    query = """
    SELECT 
        -- All camilla2 variables
        c.*,
        -- Denovo variables
        d.is_female,
        d.has_hypertension,
        d.has_diabetes,
        d.has_hyperlipidemia,
        d.has_prior_cabg,
        d.has_prior_pci,
        d.has_previous_mi,
        d.estimated_gfr,
        d.left_ventricular_ejection_fraction,
        d.clinical_presentation,
        d.target_lesion_cass_score,
        d.acc_aha_classification,
        d.calcification,
        d.is_cto,
        d.is_ostial_lesion,
        d.vascular_access,
        d.number_of_pci_vessels,
        d.number_of_dcb_treated_vessels,
        d.syntax_score,
        d.intravascular_imaging,
        d.has_lesion_preparation,
        d.used_cutting_balloon,
        d.used_scoring_balloon,
        d.used_opn_balloon,
        d.used_rotablator,
        d.used_orbital,
        d.used_shockwave,
        d.dcb_1_drug,
        d.dcb_1_diameter,
        d.dcb_1_length,
        d.dcb_1_inflation_pressure as max_dcb_pressure,
        d.dcb_1_diameter as max_dcb_diameter,
        d.dcb_1_length as max_dcb_length,
        d.number_of_dcbs_used,
        d.is_hybrid_pci,
        d.antithrombotic_therapy_at_discharge,
        d.planned_dapt_duration,
        d.on_aspirin,
        d.p2y12_inhibitor,
        d.oral_anticoagulation,
        d.is_clopidogrel,
        d.is_prasugrel,
        -- Calculate age
        EXTRACT(YEAR FROM AGE(c.date_dcb, c.dob)) as age
    FROM camilla2 c
    INNER JOIN denovo d ON c.dob = d.birth_date
    """
    
    df = pd.read_sql(query, engine)
    return df

def calculate_derived_variables(df):
    """Calculate derived variables"""
    df['date_dcb'] = pd.to_datetime(df['date_dcb'])
    df['date_fup_coro'] = pd.to_datetime(df['date_fup_coro'])
    df['days_to_followup'] = (df['date_fup_coro'] - df['date_dcb']).dt.days
    df['months_to_followup'] = df['days_to_followup'] / 30.44
    return df

# Define outcomes to analyze
OUTCOMES: List[str] = [
    'mld_late_lumen_change',      # Late lumen gain/loss
    'mufr_late_functional_change', 
    'mld_late_recoil',
    'mld_net_gain',
    'mufr_net_functional_gain'
]

# Manual variable typing
# Edit these lists to control how variables are treated
BINARY_VARS: List[str] = [
    'is_female',
    'has_hypertension',
    'has_diabetes',
    'has_prior_cabg',
    'has_hyperlipidemia',
    'has_prior_pci',
    'has_previous_mi',
    'has_lesion_preparation',
        'used_cutting_balloon',
        'used_scoring_balloon',
        'used_opn_balloon',
        'used_rotablator',
        'used_shockwave',
    'dcb_drug',
    'is_cto',
    'is_ostial_lesion'
    #'is_clopidogrel',
    #'is_prasugrel'
]

CATEGORICAL_VARS: List[str] = [
    'p2y12_inhibitor',
    'clinical_presentation',
    #'acc_aha_classification',
]

# Define continuous predictors (manual)
CONTINUOUS_VARS: List[str] = [
        'calcification',
    'age',
    'estimated_gfr',
    'left_ventricular_ejection_fraction',
    'syntax_score',
    'max_balloon_pressure',
    'planned_dapt_duration',
    'dcb_inflation_max_time',
    'dcb_max_pressure',
    'dcb_diam_to_vessel',
    'dcb_lenght_to_vessel',
    'predilatation_diam_to_vessel',
]

# Use all three groups as predictors
ALL_PREDICTORS: List[str] = list(dict.fromkeys(BINARY_VARS + CATEGORICAL_VARS + CONTINUOUS_VARS))

def classify_variable_type(series: pd.Series, var_name: str = None) -> str:
    """
    Manual classification: if variable name is listed in BINARY_VARS or CATEGORICAL_VARS.
    Defaults to continuous otherwise.
    """
    name = var_name or series.name
    if name in BINARY_VARS:
        return 'binary'
    if name in CATEGORICAL_VARS:
        return 'categorical'
    return 'continuous'

def prepare_predictors_by_type(df: pd.DataFrame, predictors: List[str]) -> Dict[str, List[str]]:
    """
    Classify predictors by their type and return dictionary
    """
    var_types = {
        'binary': [],
        'categorical': [],
        'continuous': []
    }
    
    for predictor in predictors:
        if predictor in df.columns:
            var_type = classify_variable_type(df[predictor])
            var_types[var_type].append(predictor)
    
    print("\nVariable Classification:")
    print(f"Binary variables ({len(var_types['binary'])}): {var_types['binary'][:5]}...")
    print(f"Categorical variables ({len(var_types['categorical'])}): {var_types['categorical'][:5]}...")
    print(f"Continuous variables ({len(var_types['continuous'])}): {var_types['continuous'][:5]}...")
    
    return var_types

def prepare_data_with_encoding(df: pd.DataFrame, outcome: str, predictors: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Prepare data with proper encoding for categorical variables
    """
    # Select columns
    columns_needed = [outcome] + predictors
    columns_available = [col for col in columns_needed if col in df.columns]
    
    # Create subset
    analysis_df = df[columns_available].copy()
    
    # Drop rows with missing outcome
    analysis_df = analysis_df.dropna(subset=[outcome])
    
    # Classify variables
    var_types = prepare_predictors_by_type(analysis_df, predictors)
    
    # Handle categorical variables - create dummy variables
    encoded_cols = []
    for cat_var in var_types['categorical']:
        if cat_var in analysis_df.columns:
            # Create dummy variables
            dummies = pd.get_dummies(analysis_df[cat_var], prefix=cat_var, drop_first=True)
            analysis_df = pd.concat([analysis_df, dummies], axis=1)
            encoded_cols.extend(dummies.columns.tolist())
            # Drop original categorical column
            analysis_df = analysis_df.drop(cat_var, axis=1)
    
    # Update predictor list with encoded columns
    final_predictors = var_types['binary'] + var_types['continuous'] + encoded_cols
    
    # Handle missing values
    for col in final_predictors:
        if col in analysis_df.columns:
            if analysis_df[col].dtype in ['float64', 'int64']:
                # Impute continuous/binary variables with median
                analysis_df[col].fillna(analysis_df[col].median(), inplace=True)
    
    encoding_info = {
        'var_types': var_types,
        'encoded_cols': encoded_cols,
        'final_predictors': final_predictors
    }
    
    print(f"Data prepared: {len(analysis_df)} observations with {len(final_predictors)} predictors (including encoded)")
    
    return analysis_df, encoding_info

def univariate_analysis(df: pd.DataFrame, outcome: str, predictors: List[str], 
                        outcome_type: str = 'continuous') -> pd.DataFrame:
    """
    Perform univariate analysis for each predictor
    Handles binary, categorical, and continuous predictors appropriately
    """
    results = []
    
    for predictor in predictors:
        if predictor not in df.columns:
            continue
        
        # Prepare data
        data = df[[outcome, predictor]].dropna()
        
        if len(data) < 10:
            continue
        
        # Classify predictor type
        pred_type = classify_variable_type(data[predictor])
        
        try:
            if outcome_type == 'continuous':
                # Linear regression
                if pred_type == 'categorical':
                    # For categorical, use ANOVA
                    groups = [group[outcome].values for name, group in data.groupby(predictor)]
                    if len(groups) >= 2:
                        f_stat, p_value = stats.f_oneway(*groups)
                        
                        # Calculate mean for each category
                        means = data.groupby(predictor)[outcome].mean()
                        effect_desc = f"F={f_stat:.2f}"
                        
                        results.append({
                            'Predictor': predictor,
                            'Type': pred_type,
                            'N': len(data),
                            'Test': 'ANOVA',
                            'Effect': effect_desc,
                            'P-value': p_value,
                            'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                        })
                else:
                    # For binary/continuous, use linear regression
                    X = data[predictor].values.reshape(-1, 1)
                    y = data[outcome].values
                    
                    X_sm = sm.add_constant(X)
                    model = sm.OLS(y, X_sm).fit()
                    
                    coef = model.params[1]
                    se = model.bse[1]
                    ci_lower = model.conf_int()[1][0]
                    ci_upper = model.conf_int()[1][1]
                    p_value = model.pvalues[1]
                    r_squared = model.rsquared
                    
                    results.append({
                        'Predictor': predictor,
                        'Type': pred_type,
                        'N': len(data),
                        'Test': 'Linear Reg',
                        'Coefficient': coef,
                        'SE': se,
                        '95% CI': f'({ci_lower:.4f}, {ci_upper:.4f})',
                        'R²': r_squared,
                        'P-value': p_value,
                        'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                    })
                    
            else:  # Binary outcome
                # Create binary outcome
                y = (data[outcome] > 0).astype(int)
                
                if pred_type == 'categorical':
                    # Chi-square test for categorical predictor
                    crosstab = pd.crosstab(data[predictor], y)
                    chi2, p_value, dof, expected = stats.chi2_contingency(crosstab)
                    
                    results.append({
                        'Predictor': predictor,
                        'Type': pred_type,
                        'N': len(data),
                        'N_gain': sum(y),
                        'N_loss': len(y) - sum(y),
                        'Test': 'Chi-square',
                        'Chi2': chi2,
                        'P-value': p_value,
                        'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                    })
                else:
                    # Logistic regression for binary/continuous predictor
                    X = data[predictor].values.reshape(-1, 1)
                    
                    X_sm = sm.add_constant(X)
                    model = sm.Logit(y, X_sm).fit(disp=0)
                    
                    coef = model.params[1]
                    or_value = np.exp(coef)
                    ci_lower = np.exp(model.conf_int()[1][0])
                    ci_upper = np.exp(model.conf_int()[1][1])
                    p_value = model.pvalues[1]
                    
                    # Calculate AUC
                    y_pred_prob = model.predict(X_sm)
                    auc = roc_auc_score(y, y_pred_prob)
                    
                    results.append({
                        'Predictor': predictor,
                        'Type': pred_type,
                        'N': len(data),
                        'N_gain': sum(y),
                        'N_loss': len(y) - sum(y),
                        'Test': 'Logistic Reg',
                        'OR': or_value,
                        '95% CI (OR)': f'({ci_lower:.3f}, {ci_upper:.3f})',
                        'AUC': auc,
                        'P-value': p_value,
                        'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''
                    })
                    
        except Exception as e:
            print(f"Error with {predictor}: {e}")
            continue
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('P-value')
    
    return results_df

def multivariate_analysis(df: pd.DataFrame, outcome: str, predictors: List[str],
                         outcome_type: str = 'continuous', p_threshold: float = 0.1) -> Tuple[Any, pd.DataFrame]:
    """
    Perform multivariate analysis with proper handling of variable types
    """
    # Prepare data with encoding
    data, encoding_info = prepare_data_with_encoding(df, outcome, predictors)
    
    # First, run univariate to select significant predictors
    univariate_results = univariate_analysis(df, outcome, predictors, outcome_type)
    significant_predictors = univariate_results[univariate_results['P-value'] < p_threshold]['Predictor'].tolist()
    
    if len(significant_predictors) < 1:
        print("No significant predictors found in univariate analysis")
        return None, pd.DataFrame()
    
    print(f"\nPredictors selected for multivariate model (p < {p_threshold} in univariate):")
    print(significant_predictors)
    
    # Prepare final predictor list including encoded variables
    final_predictors = []
    for pred in significant_predictors:
        if pred in encoding_info['var_types']['categorical']:
            # Find encoded columns for this categorical variable
            encoded = [col for col in encoding_info['encoded_cols'] if col.startswith(f"{pred}_")]
            final_predictors.extend(encoded)
        elif pred in data.columns:
            final_predictors.append(pred)
    
    # Remove duplicates
    final_predictors = list(set(final_predictors))
    
    # Check if we have predictors after encoding
    final_predictors = [p for p in final_predictors if p in data.columns]
    
    if len(final_predictors) == 0:
        print("No predictors available after encoding")
        return None, pd.DataFrame()
    
    print(f"Final predictors after encoding: {final_predictors}")
    
    # Prepare data - clean column names to avoid formula parsing issues
    analysis_data = data[[outcome] + final_predictors].copy()
    
    # Clean column names for formula - replace problematic characters
    clean_names = {}
    for col in analysis_data.columns:
        clean_name = col.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        clean_names[col] = clean_name
        
    analysis_data = analysis_data.rename(columns=clean_names)
    outcome_clean = clean_names[outcome]
    final_predictors_clean = [clean_names[p] for p in final_predictors]
    
    # Drop missing values
    analysis_data = analysis_data.dropna()
    
    # Check sample size
    n_obs = len(analysis_data)
    n_predictors = len(final_predictors_clean)
    
    if outcome_type == 'continuous':
        # Linear regression
        if n_obs < 10 * n_predictors:
            print(f"Warning: Sample size ({n_obs}) may be insufficient for {n_predictors} predictors")
            # Reduce predictors if sample too small
            if n_obs < 5 * n_predictors:
                max_predictors = max(1, n_obs // 10)
                print(f"Reducing predictors to top {max_predictors} based on univariate p-values")
                # Keep only top predictors
                top_preds = univariate_results[univariate_results['P-value'] < p_threshold].head(max_predictors)['Predictor'].tolist()
                # Update final predictors
                final_predictors_clean = [clean_names.get(p, p) for p in top_preds if clean_names.get(p, p) in analysis_data.columns]
                n_predictors = len(final_predictors_clean)
        
        # Build formula using clean names
        formula = f"{outcome_clean} ~ " + " + ".join(final_predictors_clean)
        
        try:
            # Fit model
            model = smf.ols(formula=formula, data=analysis_data).fit()
            
            # Create results table - map back to original names
            results = []
            param_names = model.params.index.tolist()
            
            # Create reverse mapping
            reverse_clean_names = {v: k for k, v in clean_names.items()}
            
            for i, param in enumerate(param_names):
                # Get original name if available
                original_name = reverse_clean_names.get(param, param)
                results.append({
                    'Variable': original_name,
                    'Coefficient': model.params[i],
                    'SE': model.bse[i],
                    '95% CI': f'({model.conf_int().iloc[i, 0]:.4f}, {model.conf_int().iloc[i, 1]:.4f})',
                    'P-value': model.pvalues[i],
                    'Significant': '***' if model.pvalues[i] < 0.001 else '**' if model.pvalues[i] < 0.01 else '*' if model.pvalues[i] < 0.05 else ''
                })
            
            results_df = pd.DataFrame(results)
            
            # Model diagnostics
            print("\nModel Summary:")
            print(f"N observations: {n_obs}")
            print(f"R-squared: {model.rsquared:.4f}")
            print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
            print(f"AIC: {model.aic:.2f}")
            print(f"BIC: {model.bic:.2f}")
            print(f"F-statistic: {model.fvalue:.4f} (p={model.f_pvalue:.4e})")
            
        except Exception as e:
            print(f"Error fitting model: {e}")
            return None, pd.DataFrame()
        
    else:  # Binary outcome
        # Create binary outcome
        analysis_data['outcome_binary'] = (analysis_data[outcome_clean] > 0).astype(int)
        
        # Check events
        n_events = min(sum(analysis_data['outcome_binary']), 
                      len(analysis_data) - sum(analysis_data['outcome_binary']))
        
        if n_events < 10 * n_predictors:
            print(f"Warning: Too few events ({n_events}) for {n_predictors} predictors")
            # Reduce predictors if too few events
            if n_events < 5 * n_predictors:
                max_predictors = max(1, n_events // 10)
                print(f"Reducing predictors to top {max_predictors} based on univariate p-values")
                # Keep only top predictors
                top_preds = univariate_results[univariate_results['P-value'] < p_threshold].head(max_predictors)['Predictor'].tolist()
                # Update final predictors
                final_predictors_clean = [clean_names.get(p, p) for p in top_preds if clean_names.get(p, p) in analysis_data.columns]
                n_predictors = len(final_predictors_clean)
        
        # Build formula using clean names
        formula = f"outcome_binary ~ " + " + ".join(final_predictors_clean)
        
        try:
            # Fit model
            model = smf.logit(formula=formula, data=analysis_data).fit(disp=0)
            
            # Create results table
            results = []
            param_names = model.params.index.tolist()
            for i, param in enumerate(param_names):
                clean_name = param.replace('`', '')
                results.append({
                    'Variable': clean_name,
                    'Coefficient': model.params[i],
                    'OR': np.exp(model.params[i]),
                    'SE': model.bse[i],
                    '95% CI (OR)': f'({np.exp(model.conf_int().iloc[i, 0]):.3f}, {np.exp(model.conf_int().iloc[i, 1]):.3f})',
                    'P-value': model.pvalues[i],
                    'Significant': '***' if model.pvalues[i] < 0.001 else '**' if model.pvalues[i] < 0.01 else '*' if model.pvalues[i] < 0.05 else ''
                })
            
            results_df = pd.DataFrame(results)
            
            # Calculate overall model performance
            y_true = analysis_data['outcome_binary']
            y_pred_prob = model.predict(analysis_data[final_predictors])
            auc = roc_auc_score(y_true, y_pred_prob)
            
            print("\nModel Summary:")
            print(f"N observations: {n_obs}")
            print(f"N with gain: {sum(y_true)}")
            print(f"N with loss: {len(y_true) - sum(y_true)}")
            print(f"AUC: {auc:.3f}")
            print(f"Pseudo R-squared: {model.prsquared:.4f}")
            print(f"Log-Likelihood: {model.llf:.2f}")
            print(f"AIC: {model.aic:.2f}")
            print(f"BIC: {model.bic:.2f}")
            
        except Exception as e:
            print(f"Error fitting multivariate model: {e}")
            return None, pd.DataFrame()
    
    # Check for multicollinearity
    if len(final_predictors) > 1 and outcome_type == 'continuous':
        X = analysis_data[final_predictors]
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        try:
            vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
            print("\nVariance Inflation Factors:")
            print(vif_data[vif_data['VIF'] < 100])  # Filter out infinite VIFs
            vif_data.to_csv(f'stats/results/vif_{outcome}_{outcome_type}.csv', index=False)
        except:
            print("Could not calculate VIF")
    
    return model, results_df

def create_forest_plot(results_df: pd.DataFrame, outcome: str, analysis_type: str, top_n: int = 15):
    """Create forest plot showing effect sizes with confidence intervals"""
    
    # Filter to significant results and top N
    sig_df = results_df[results_df['P-value'] < 0.05].head(top_n)
    
    if sig_df.empty:
        print(f"No significant predictors for {outcome}")
        return
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(sig_df) * 0.5)))
    
    if 'OR' in sig_df.columns:  # Binary outcome
        # Keep only rows with valid OR and CI
        plot_df = sig_df.copy()
        if 'OR' in plot_df.columns:
            plot_df = plot_df[plot_df['OR'].notna()]
        if '95% CI (OR)' in plot_df.columns:
            plot_df = plot_df[plot_df['95% CI (OR)'].notna()]

        or_values = []
        ci_lower = []
        ci_upper = []
        predictor_labels = []
        sig_markers = []

        for _, row in plot_df.iterrows():
            ci_val = row.get('95% CI (OR)')
            lower = upper = None
            if isinstance(ci_val, str):
                s = ci_val.strip().strip('()')
                parts = [p.strip() for p in s.split(',')]
                if len(parts) == 2:
                    try:
                        lower, upper = float(parts[0]), float(parts[1])
                    except Exception:
                        lower = upper = None
            elif isinstance(ci_val, (list, tuple, np.ndarray)) and len(ci_val) == 2:
                try:
                    lower, upper = float(ci_val[0]), float(ci_val[1])
                except Exception:
                    lower = upper = None

            if row.get('OR') is not None and lower is not None and upper is not None:
                or_values.append(row['OR'])
                ci_lower.append(lower)
                ci_upper.append(upper)
                predictor_labels.append(row['Predictor'])
                sig_markers.append(row.get('Significant', ''))

        if len(or_values) == 0:
            print(f"No plottable ORs for {outcome}")
            return

        y_pos = np.arange(len(or_values))

        # Plot on log scale
        ax.scatter(or_values, y_pos, s=50, color='blue', zorder=3)

        # Add CI lines
        for i, (lower, upper) in enumerate(zip(ci_lower, ci_upper)):
            ax.plot([lower, upper], [i, i], 'b-', linewidth=2, alpha=0.7)

        ax.axvline(x=1, color='red', linestyle='--', alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel('Odds Ratio (95% CI)', fontsize=12)
        title_suffix = "Odds Ratios"
        
    elif 'Coefficient' in sig_df.columns:  # Continuous outcome
        # Keep only rows with valid Coefficient and CI
        plot_df = sig_df.copy()
        if 'Coefficient' in plot_df.columns:
            plot_df = plot_df[plot_df['Coefficient'].notna()]
        if '95% CI' in plot_df.columns:
            plot_df = plot_df[plot_df['95% CI'].notna()]

        coef_values = []
        ci_lower = []
        ci_upper = []
        predictor_labels = []
        sig_markers = []

        for _, row in plot_df.iterrows():
            ci_val = row.get('95% CI')
            lower = upper = None
            if isinstance(ci_val, str):
                s = ci_val.strip().strip('()')
                parts = [p.strip() for p in s.split(',')]
                if len(parts) == 2:
                    try:
                        lower, upper = float(parts[0]), float(parts[1])
                    except Exception:
                        lower = upper = None
            elif isinstance(ci_val, (list, tuple, np.ndarray)) and len(ci_val) == 2:
                try:
                    lower, upper = float(ci_val[0]), float(ci_val[1])
                except Exception:
                    lower = upper = None

            if row.get('Coefficient') is not None and lower is not None and upper is not None:
                coef_values.append(row['Coefficient'])
                ci_lower.append(lower)
                ci_upper.append(upper)
                predictor_labels.append(row['Predictor'])
                sig_markers.append(row.get('Significant', ''))

        if len(coef_values) == 0:
            print(f"No plottable coefficients for {outcome}")
            return

        y_pos = np.arange(len(coef_values))

        # Plot
        ax.scatter(coef_values, y_pos, s=50, color='green', zorder=3)

        # Add CI lines
        for i, (lower, upper) in enumerate(zip(ci_lower, ci_upper)):
            ax.plot([lower, upper], [i, i], 'g-', linewidth=2, alpha=0.7)

        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Coefficient (95% CI)', fontsize=12)
        title_suffix = "Coefficients"
    
    # Set labels using the plotted subset
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictor_labels)
    ax.set_ylabel('Predictor', fontsize=12)
    ax.set_title(f'Significant Predictors of {outcome}\n{title_suffix} with 95% Confidence Intervals', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add p-value annotations
    for i, sig_marker in enumerate(sig_markers):
        ax.text(ax.get_xlim()[1] * 0.95, i, sig_marker,
                ha='right', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    filename = f'stats/plots/forest_{outcome}_{analysis_type}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    print(f"Forest plot saved to {filename}")

def create_predictor_comparison_plot(df: pd.DataFrame, outcome: str):
    """Create box plots comparing outcome by categorical predictors"""
    
    # Select categorical predictors
    categorical_predictors = []
    for pred in ALL_PREDICTORS:
        if pred in df.columns:
            if classify_variable_type(df[pred]) in ['binary', 'categorical']:
                categorical_predictors.append(pred)
    
    # Select top predictors based on univariate p-values
    univariate_results = univariate_analysis(df, outcome, categorical_predictors, 'continuous')
    
    if univariate_results.empty:
        return
    
    top_predictors = univariate_results[univariate_results['P-value'] < 0.05].head(6)['Predictor'].tolist()
    
    if not top_predictors:
        top_predictors = univariate_results.head(6)['Predictor'].tolist()
    
    n_plots = len(top_predictors)
    if n_plots == 0:
        return
    
    # Create subplots
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, predictor in enumerate(top_predictors):
        ax = axes[i]
        
        # Prepare data
        plot_data = df[[predictor, outcome]].dropna()
        
        # Create box plot
        unique_vals = sorted(plot_data[predictor].unique())
        data_by_group = [plot_data[plot_data[predictor] == val][outcome] for val in unique_vals]
        
        bp = ax.boxplot(data_by_group, labels=[f'{val}' for val in unique_vals], patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_vals)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Add sample sizes
        for j, val in enumerate(unique_vals):
            n = len(data_by_group[j])
            ax.text(j+1, ax.get_ylim()[0], f'n={n}', ha='center', fontsize=8)
        
        ax.set_xlabel(predictor, fontsize=10)
        ax.set_ylabel(outcome, fontsize=10)
        ax.set_title(f'{outcome} by {predictor}', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0 if outcome can be negative
        if plot_data[outcome].min() < 0:
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Remove empty subplots
    for i in range(n_plots, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(f'Outcome Distribution by Categorical Predictors\n{outcome}', fontsize=14)
    plt.tight_layout()
    
    filename = f'stats/plots/boxplots_{outcome}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    print(f"Box plots saved to {filename}")

def create_correlation_matrix(df: pd.DataFrame, outcome: str):
    """Create correlation matrix for continuous predictors with outcome"""
    
    # Select continuous predictors
    continuous_predictors = []
    for pred in ALL_PREDICTORS:
        if pred in df.columns:
            if classify_variable_type(df[pred]) == 'continuous':
                continuous_predictors.append(pred)
    
    # Add outcome
    vars_to_correlate = [outcome] + continuous_predictors[:20]  # Limit to 20 for readability
    
    # Calculate correlation matrix
    corr_data = df[vars_to_correlate].dropna()
    corr_matrix = corr_data.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
               cmap='coolwarm', center=0, square=True,
               linewidths=0.5, cbar_kws={"shrink": 0.8},
               vmin=-1, vmax=1)
    
    plt.title(f'Correlation Matrix - {outcome} with Continuous Predictors', fontsize=14)
    plt.tight_layout()
    
    filename = f'stats/plots/correlation_{outcome}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    print(f"Correlation matrix saved to {filename}")
    
    # Print top correlations with outcome
    outcome_corr = corr_matrix[outcome].drop(outcome).sort_values(ascending=False)
    
    print(f"\nTop correlations with {outcome}:")
    print(outcome_corr.head(10))
    
    return corr_matrix

def create_summary_tables(all_results: Dict):
    """Create summary tables of significant predictors across all analyses"""
    
    summary_continuous = []
    summary_binary = []
    
    for outcome, results in all_results.items():
        # Continuous outcome predictors
        if 'univariate_continuous' in results and not results['univariate_continuous'].empty:
            sig_cont = results['univariate_continuous'][results['univariate_continuous']['P-value'] < 0.05]
            for _, row in sig_cont.iterrows():
                summary_continuous.append({
                    'Outcome': outcome,
                    'Predictor': row['Predictor'],
                    'Predictor Type': row.get('Type', 'unknown'),
                    'Effect': row.get('Coefficient', row.get('Effect', 'N/A')),
                    'P-value': row['P-value'],
                    'Significance': row['Significant']
                })
        
        # Binary outcome predictors  
        if 'univariate_binary' in results and not results['univariate_binary'].empty:
            sig_bin = results['univariate_binary'][results['univariate_binary']['P-value'] < 0.05]
            for _, row in sig_bin.iterrows():
                summary_binary.append({
                    'Outcome': f"{outcome} (gain vs loss)",
                    'Predictor': row['Predictor'],
                    'Predictor Type': row.get('Type', 'unknown'),
                    'OR': row.get('OR', row.get('Chi2', 'N/A')),
                    'P-value': row['P-value'],
                    'Significance': row['Significant']
                })
    
    # Save summaries
    if summary_continuous:
        cont_df = pd.DataFrame(summary_continuous)
        cont_df = cont_df.sort_values(['Outcome', 'P-value'])
        cont_df.to_csv('stats/results/summary_predictors_continuous.csv', index=False)
        
        print("\n" + "="*80)
        print("SUMMARY: TOP PREDICTORS OF CONTINUOUS OUTCOMES")
        print("="*80)
        print(cont_df.head(20).to_string(index=False))
    
    if summary_binary:
        bin_df = pd.DataFrame(summary_binary)
        bin_df = bin_df.sort_values(['Outcome', 'P-value'])
        bin_df.to_csv('stats/results/summary_predictors_binary.csv', index=False)
        
        print("\n" + "="*80)
        print("SUMMARY: TOP PREDICTORS OF BINARY OUTCOMES (GAIN vs LOSS)")
        print("="*80)
        print(bin_df.head(20).to_string(index=False))

def perform_complete_analysis(df: pd.DataFrame, outcome: str) -> Dict:
    """
    Complete analysis pipeline for one outcome:
    1. Univariate analysis (continuous outcome)
    2. Univariate analysis (binary outcome: gain vs loss)
    3. Multivariate analysis (continuous)
    4. Multivariate analysis (binary)
    """
    
    print("\n" + "="*80)
    print(f"COMPLETE ANALYSIS FOR: {outcome}")
    print("="*80)
    
    # Check if outcome exists
    if outcome not in df.columns:
        print(f"Outcome {outcome} not found in dataset")
        return {}
    
    # Get available predictors
    available_predictors = [p for p in ALL_PREDICTORS if p in df.columns]
    
    # Basic statistics
    print(f"\nOutcome Statistics:")
    print(f"  N total: {len(df)}")
    print(f"  N with outcome data: {df[outcome].notna().sum()}")
    print(f"  Mean ± SD: {df[outcome].mean():.3f} ± {df[outcome].std():.3f}")
    print(f"  Median (IQR): {df[outcome].median():.3f} ({df[outcome].quantile(0.25):.3f}, {df[outcome].quantile(0.75):.3f})")
    print(f"  % with gain (>0): {(df[outcome] > 0).mean()*100:.1f}%")
    
    results = {}
    
    # 1. UNIVARIATE ANALYSIS - CONTINUOUS OUTCOME
    print("\n" + "-"*60)
    print("1. UNIVARIATE ANALYSIS - CONTINUOUS OUTCOME")
    print("-"*60)
    
    univariate_cont = univariate_analysis(df, outcome, available_predictors, 'continuous')
    print(f"\nSignificant predictors (p<0.05): {sum(univariate_cont['P-value'] < 0.05)}/{len(univariate_cont)}")
    
    # Show top results
    if not univariate_cont.empty:
        print("\nTop 10 predictors:")
        print(univariate_cont.head(10)[['Predictor', 'Type', 'P-value', 'Significant']].to_string(index=False))
        univariate_cont.to_csv(f'stats/results/univariate_continuous_{outcome}.csv', index=False)
    
    results['univariate_continuous'] = univariate_cont
    
    # 2. UNIVARIATE ANALYSIS - BINARY OUTCOME
    print("\n" + "-"*60)
    print("2. UNIVARIATE ANALYSIS - BINARY OUTCOME (GAIN vs LOSS)")
    print("-"*60)
    
    univariate_bin = univariate_analysis(df, outcome, available_predictors, 'binary')
    print(f"\nSignificant predictors (p<0.05): {sum(univariate_bin['P-value'] < 0.05)}/{len(univariate_bin)}")
    
    # Show top results
    if not univariate_bin.empty:
        print("\nTop 10 predictors:")
        print(univariate_bin.head(10)[['Predictor', 'Type', 'P-value', 'Significant']].to_string(index=False))
        univariate_bin.to_csv(f'stats/results/univariate_binary_{outcome}.csv', index=False)
    
    results['univariate_binary'] = univariate_bin
    
    # 3. MULTIVARIATE ANALYSIS - CONTINUOUS
    print("\n" + "-"*60)
    print("3. MULTIVARIATE ANALYSIS - CONTINUOUS OUTCOME")
    print("-"*60)
    
    multi_model_cont, multi_results_cont = multivariate_analysis(df, outcome, available_predictors, 'continuous')
    
    if not multi_results_cont.empty:
        print("\nMultivariate model results:")
        print(multi_results_cont.to_string(index=False))
        multi_results_cont.to_csv(f'stats/results/multivariate_continuous_{outcome}.csv', index=False)
    
    results['multivariate_continuous'] = (multi_model_cont, multi_results_cont)
    
    # 4. MULTIVARIATE ANALYSIS - BINARY
    print("\n" + "-"*60)
    print("4. MULTIVARIATE ANALYSIS - BINARY OUTCOME")
    print("-"*60)
    
    multi_model_bin, multi_results_bin = multivariate_analysis(df, outcome, available_predictors, 'binary')
    
    if not multi_results_bin.empty:
        print("\nMultivariate model results:")
        print(multi_results_bin.to_string(index=False))
        multi_results_bin.to_csv(f'stats/results/multivariate_binary_{outcome}.csv', index=False)
    
    results['multivariate_binary'] = (multi_model_bin, multi_results_bin)
    
    # 5. CREATE VISUALIZATIONS
    print("\n" + "-"*60)
    print("5. CREATING VISUALIZATIONS")
    print("-"*60)
    
    # Forest plots
    if not univariate_cont.empty:
        create_forest_plot(univariate_cont, outcome, 'continuous')
    
    if not univariate_bin.empty:
        create_forest_plot(univariate_bin, outcome, 'binary')
    
    # Box plots for categorical predictors
    create_predictor_comparison_plot(df, outcome)
    
    # Correlation matrix
    create_correlation_matrix(df, outcome)
    
    return results

def main():
    """Main execution function"""
    
    print("="*80)
    print("PREDICTIVE ANALYSIS: IDENTIFYING PREDICTORS OF LATE LUMEN AND PHYSIOLOGICAL CHANGES")
    print("="*80)
    
    # Create directories
    create_directories()
    
    # Connect to database
    engine = connect_to_db()
    if engine is None:
        print("Failed to connect to database. Please check connection parameters.")
        return
    
    # Load and prepare data
    print("\nLoading data...")
    df = load_data(engine)
    df = calculate_derived_variables(df)
    print(f"Loaded {len(df)} observations")
    
    # Analyze each outcome
    all_results = {}
    
    for outcome in OUTCOMES:
        results = perform_complete_analysis(df, outcome)
        all_results[outcome] = results
    
    # Create summary tables
    create_summary_tables(all_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print("\nKey Questions Answered:")
    print("1. What are the predictors of late lumen gain/loss?")
    print("2. What are the predictors of late physiological gain/loss?")
    print("3. Which predictors remain significant after multivariate adjustment?")
    print("4. How well can we predict who will have gain vs loss?")
    
    print("\nResults saved to:")
    print("- Detailed tables: stats/results/")
    print("- Visualizations: stats/plots/")
    
    # Final summary of most important predictors
    print("\n" + "="*80)
    print("MOST IMPORTANT PREDICTORS (p<0.001 in univariate)")
    print("="*80)
    
    for outcome, results in all_results.items():
        print(f"\n{outcome}:")
        if 'univariate_continuous' in results and not results['univariate_continuous'].empty:
            top = results['univariate_continuous'][results['univariate_continuous']['P-value'] < 0.001]
            if not top.empty:
                print("  Continuous outcome predictors:")
                for _, row in top.head(5).iterrows():
                    print(f"    - {row['Predictor']} (p={row['P-value']:.4f})")
        
        if 'univariate_binary' in results and not results['univariate_binary'].empty:
            top = results['univariate_binary'][results['univariate_binary']['P-value'] < 0.001]
            if not top.empty:
                print("  Binary outcome predictors (gain vs loss):")
                for _, row in top.head(5).iterrows():
                    if 'OR' in row:
                        print(f"    - {row['Predictor']} (OR={row['OR']:.2f}, p={row['P-value']:.4f})")
                    else:
                        print(f"    - {row['Predictor']} (p={row['P-value']:.4f})")
    
    return all_results

if __name__ == "__main__":
    results = main()