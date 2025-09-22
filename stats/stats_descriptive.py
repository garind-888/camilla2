import pandas as pd
import numpy as np
from scipy import stats
import psycopg2
from sqlalchemy import create_engine
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Database connection parameters
DB_CONFIG = {
    'host': 'localhost',  # Update with your host
    'database': 'dcb',
    'user': 'doriangarin',  # Update with your username
    'password': '96349130dG!',  # Update with your password
    'port': 5432
}

# Create directories for outputs
def create_directories():
    """Create directory for saving tabular results"""
    os.makedirs('stats/baseline', exist_ok=True)
    print("Created directory: stats/baseline")

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
        -- Camilla2 table variables
        c.dob, c.date_dcb, c.date_fup_coro,
        c.tvf_fup_coro, c.tlf_fup_coro, c.tvf_in_between,
        
        -- Initial pre variables
        c.init_pre_lesion_length, c.init_pre_ref_vessel_diam, c.init_pre_min_vessel_diam,
        c.init_pre_mld, c.init_pre_mufr,
        
        -- Initial post variables
        c.init_post_lesion_length, c.init_post_ref_vessel_diam,
        c.init_post_min_vessel_diam, c.init_post_mld,
        c.init_post_mufr, c.init_post_flow, c.init_post_as, c.init_post_ds,
        
        -- Follow-up pre variables
        c.is_residual_stenosis_at_fup,
        c.fup_pre_lesion_length, c.fup_pre_ref_vessel_diam, c.fup_pre_min_vessel_diam,
        c.fup_pre_mld, c.fup_pre_mufr,
        c.fup_pre_flow, c.fup_pre_as, c.fup_pre_df,
        
        -- Follow-up post variables
        c.fup_post_lesion_length, c.fup_post_ref_vessel_diam, c.fup_post_min_vessel_diam,
        c.fup_post_mld, c.fup_post_mufr,
        c.fup_post_flow, c.fup_post_as, c.fup_post_ds,
        
        -- New calculated metrics from SQL script
        c.mld_late_lumen_change, c.mld_acute_recoil, c.mld_late_recoil,
        c.mld_acute_gain, c.mld_net_gain,
        c.mufr_late_functional_change, c.mufr_net_functional_gain, c.mufr_acute_functional_gain,
        c.dcb_balloon_diameter, c.dcb_balloon_length,
        c.dcb_inflation_max_time, c.dcb_max_pressure, c.dcb_drug,
        c.dcb_diam_to_vessel, c.dcb_lenght_to_vessel,
        -- Auto-calculated fields (from denovo)
        d.max_dcb_length, d.max_dcb_diameter, d.max_dcb_pressure,
        d.age,
        d.max_balloon_length, d.max_balloon_diameter, d.max_balloon_pressure,
        
        -- Vessel location variables from camilla2
        c.is_lad, c.is_rca, c.is_lcx, c.is_bifurcation,
        
        -- Denovo table variables - Patient characteristics
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
        
        -- Denovo table variables - Procedural characteristics
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
        d.dcb_1_inflation_pressure,
        d.number_of_dcbs_used,
        d.is_hybrid_pci,
        d.planned_dapt_duration,
        d.on_aspirin,
        d.p2y12_inhibitor,
        d.oral_anticoagulation
        
    FROM camilla2 c
    INNER JOIN denovo d ON c.dob = d.birth_date
    """
    
    df = pd.read_sql(query, engine)
    return df

def load_metrics_data(engine):
    """Load lesion evolution metrics and outcomes directly from camilla2 (no join)"""
    query = """
    SELECT 
        -- Dates for derived variables
        dob, date_dcb, date_fup_coro,
        
        -- MLD metrics across timepoints and derived
        init_pre_mld, init_post_mld, fup_pre_mld,
        mld_acute_gain, mld_late_lumen_change, mld_net_gain,
        mld_acute_recoil, mld_late_recoil,
        
        -- Lesion length across timepoints
        init_pre_lesion_length, init_post_lesion_length, fup_pre_lesion_length, fup_post_lesion_length,

        -- Reference vessel diameter across timepoints
        init_pre_ref_vessel_diam, init_post_ref_vessel_diam, fup_pre_ref_vessel_diam, fup_post_ref_vessel_diam,

        -- MLA/μFR metrics
        init_pre_mla, init_post_mla, fup_pre_mla,
        init_pre_mufr, init_post_mufr, fup_pre_mufr,
        mufr_acute_functional_gain, mufr_late_functional_change, mufr_net_functional_gain,
        
        -- Other lesion variables that may be used in tests
        init_pre_lesion_length as init_pre_lesion_length_dup,
        
        -- Outcomes
        tvf_fup_coro, tlf_fup_coro, is_residual_stenosis_at_fup
    FROM camilla2
    """
    df = pd.read_sql(query, engine)
    return df

def calculate_derived_variables(df):
    """Calculate derived variables like age, follow-up time"""
    # Calculate age at DCB
    df['date_dcb'] = pd.to_datetime(df['date_dcb'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['date_fup_coro'] = pd.to_datetime(df['date_fup_coro'])
    
    # Prefer DB-provided age if available, otherwise compute
    if 'age' not in df.columns or df['age'].isna().all():
        df['age_at_dcb'] = (df['date_dcb'] - df['dob']).dt.days / 365.25
        df['age'] = df['age_at_dcb']
    else:
        # Ensure numeric
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['days_to_followup'] = (df['date_fup_coro'] - df['date_dcb']).dt.days
    df['months_to_followup'] = df['days_to_followup'] / 30.44
    
    return df

def format_continuous_variable(data, name):
    """Format continuous variable as mean±SD and median (IQR)"""
    if data.dropna().empty:
        return {
            'Variable': name,
            'N': 0,
            'Mean ± SD': '—',
            'Median (IQR)': '—'
        }
    
    mean = data.mean()
    std = data.std()
    median = data.median()
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    n = data.notna().sum()
    
    return {
        'Variable': name,
        'N': n,
        'Mean ± SD': f"{mean:.2f} ± {std:.2f}",
        'Median (IQR)': f"{median:.2f} ({q1:.2f}-{q3:.2f})"
    }

def format_categorical_variable(data, name, value_labels=None):
    """Format categorical or boolean variable as n (%)"""
    total = len(data.dropna())
    if total == 0:
        return [{
            'Variable': name,
            'Category': 'No data',
            'N (%)': '0 (0.0%)'
        }]
    
    value_counts = data.value_counts()
    results = []
    
    for value, count in value_counts.items():
        percentage = (count / total) * 100
        if value_labels and value in value_labels:
            label = value_labels[value]
        elif pd.isna(value):
            label = 'Missing'
        elif isinstance(value, bool) or value in [0, 1]:
            label = 'Yes' if value else 'No'
        else:
            label = str(value)
        
        results.append({
            'Variable': name if len(results) == 0 else '',
            'Category': label,
            'N (%)': f"{count} ({percentage:.1f}%)"
        })
    
    return results

def create_baseline_characteristics_table(df):
    """Create baseline patient characteristics table"""
    print("\n" + "=" * 80)
    print("BASELINE PATIENT CHARACTERISTICS")
    print("=" * 80)
    print(f"Total patients: {len(df)}")
    print("-" * 80)
    
    # Continuous variables
    continuous_vars = [
        ('age', 'Age (years)'),
        ('estimated_gfr', 'eGFR (mL/min/1.73m²)'),
        ('left_ventricular_ejection_fraction', 'LVEF (%)')
    ]
    
    continuous_results = []
    for var, label in continuous_vars:
        if var in df.columns:
            continuous_results.append(format_continuous_variable(df[var], label))
    
    continuous_df = pd.DataFrame(continuous_results)
    print("\nContinuous Variables:")
    print(continuous_df.to_string(index=False))
    
    # Categorical/Boolean variables
    print("\nCategorical Variables:")
    categorical_vars = [
        ('is_female', 'Female sex'),
        ('has_hypertension', 'Hypertension'),
        ('has_diabetes', 'Diabetes mellitus'),
        ('has_hyperlipidemia', 'Hyperlipidemia'),
        ('has_prior_cabg', 'Prior CABG'),
        ('has_prior_pci', 'Prior PCI'),
        ('has_previous_mi', 'Previous MI')
    ]
    
    categorical_results = []
    for var, label in categorical_vars:
        if var in df.columns:
            categorical_results.extend(format_categorical_variable(df[var], label))
    
    # Clinical presentation
    if 'clinical_presentation' in df.columns:
        presentation_labels = {
            0: 'CCS',
            1: 'UA',
            2: 'NSTEMI',
            3: 'STEMI'
        }
        categorical_results.extend(
            format_categorical_variable(df['clinical_presentation'], 
                                       'Clinical presentation', 
                                       presentation_labels)
        )
    
    categorical_df = pd.DataFrame(categorical_results)
    print(categorical_df.to_string(index=False))
    
    # Save to CSV
    baseline_table = pd.concat([continuous_df, categorical_df], ignore_index=True)
    baseline_table.to_csv('stats/results/baseline/baseline_characteristics.csv', index=False)
    
    return baseline_table

def create_procedural_characteristics_table(df):
    """Create procedural characteristics table"""
    print("\n" + "=" * 80)
    print("PROCEDURAL CHARACTERISTICS")
    print("=" * 80)
    print("-" * 80)
    
    # Continuous procedural variables
    # Prefer auto-calculated maxima when available, with fallback to legacy fields
    continuous_vars = [
        ('init_pre_lesion_length', 'Baseline lesion length (mm)'),
        ('init_pre_ref_vessel_diam', 'Baseline reference diameter (mm)'),
        ('init_pre_mld', 'Baseline MLD (mm)'),
        ('syntax_score', 'SYNTAX score'),
        ('max_dcb_diameter', 'Max DCB diameter (mm)'),
        ('max_dcb_length', 'Max DCB length (mm)'),
        ('max_dcb_pressure', 'Max DCB inflation pressure (atm)'),
        ('max_balloon_diameter', 'Max pre-dilation balloon diameter (mm)'),
        ('max_balloon_pressure', 'Max pre-dilation balloon pressure (atm)'),
        ('planned_dapt_duration', 'Planned DAPT duration (months)'),
        # Additional fields from camilla2 if available
        ('dcb_balloon_diameter', 'DCB diameter (mm)'),
        ('dcb_balloon_length', 'DCB length (mm)'),
        ('dcb_inflation_max_time', 'DCB max inflation time (s)'),
        ('dcb_max_pressure', 'DCB max pressure (atm)'),
        ('dcb_diam_to_vessel', 'DCB diameter-to-vessel ratio'),
        ('dcb_lenght_to_vessel', 'DCB length-to-vessel ratio'),
        ('max_balloon_length', 'Max pre-dilation balloon length (mm)'),
        ('predilatation_lenght_to_vessel', 'Predilatation length-to-vessel ratio'),
        ('predilatation_diam_to_vessel', 'Predilatation diameter-to-vessel ratio')
    ]
    
    continuous_results = []
    for var, label in continuous_vars:
        series = None
        if var in df.columns:
            series = df[var]
        # Fallbacks for legacy DCB fields if max_* not present
        if series is None and var == 'max_dcb_diameter' and 'dcb_1_diameter' in df.columns:
            series = df['dcb_1_diameter']
        if series is None and var == 'max_dcb_length' and 'dcb_1_length' in df.columns:
            series = df['dcb_1_length']
        if series is None and var == 'max_dcb_pressure' and 'dcb_1_inflation_pressure' in df.columns:
            series = df['dcb_1_inflation_pressure']
        if series is not None:
            continuous_results.append(format_continuous_variable(series, label))
    
    continuous_df = pd.DataFrame(continuous_results)
    print("\nContinuous Variables:")
    print(continuous_df.to_string(index=False))
    
    # Categorical/Boolean procedural variables
    print("\nCategorical Variables:")
    categorical_results = []
    
    # Vessel location
    vessel_vars = [
        ('is_lad', 'LAD'),
        ('is_rca', 'RCA'),
        ('is_lcx', 'LCX')
    ]
    
    print("\nTarget Vessel:")
    for var, label in vessel_vars:
        if var in df.columns:
            categorical_results.extend(format_categorical_variable(df[var], label))
    
    # Lesion characteristics
    lesion_vars = [
        ('is_bifurcation', 'Bifurcation lesion'),
        ('calcification', 'Calcification present'),
        ('is_cto', 'Chronic total occlusion'),
        ('is_ostial_lesion', 'Ostial lesion'),
        ('intravascular_imaging', 'Intravascular imaging used'),
        ('has_lesion_preparation', 'Lesion preparation performed')
    ]
    
    for var, label in lesion_vars:
        if var in df.columns:
            categorical_results.extend(format_categorical_variable(df[var], label))
    
    # Preparation devices
    prep_vars = [
        ('used_cutting_balloon', 'Cutting balloon'),
        ('used_scoring_balloon', 'Scoring balloon'),
        ('used_opn_balloon', 'OPN balloon'),
        ('used_rotablator', 'Rotablator'),
        ('used_orbital', 'Orbital atherectomy'),
        ('used_shockwave', 'Shockwave')
    ]
    
    print("\nLesion Preparation Devices:")
    for var, label in prep_vars:
        if var in df.columns:
            categorical_results.extend(format_categorical_variable(df[var], label))
    
    # DCB drug type
    if 'dcb_1_drug' in df.columns or 'dcb_drug' in df.columns:
        drug_labels = {
            1: 'Paclitaxel',
            2: 'Sirolimus',
            3: 'Other'
        }
        # Prefer camilla2 dcb_drug if present, else fall back to denovo dcb_1_drug
        if 'dcb_drug' in df.columns:
            categorical_results.extend(
                format_categorical_variable(df['dcb_drug'], 'DCB drug type', drug_labels)
            )
        else:
            categorical_results.extend(
                format_categorical_variable(df['dcb_1_drug'], 'DCB drug type', drug_labels)
            )
    
    # Number of vessels/DCBs
    if 'number_of_pci_vessels' in df.columns:
        categorical_results.append({
            'Variable': 'Number of PCI vessels',
            'Category': 'Mean ± SD',
            'N (%)': f"{df['number_of_pci_vessels'].mean():.1f} ± {df['number_of_pci_vessels'].std():.1f}"
        })
    
    if 'number_of_dcbs_used' in df.columns:
        categorical_results.append({
            'Variable': 'Number of DCBs used',
            'Category': 'Mean ± SD',
            'N (%)': f"{df['number_of_dcbs_used'].mean():.1f} ± {df['number_of_dcbs_used'].std():.1f}"
        })
    
    # Hybrid PCI
    if 'is_hybrid_pci' in df.columns:
        categorical_results.extend(format_categorical_variable(df['is_hybrid_pci'], 'Hybrid PCI'))
    
    # Antithrombotic therapy
    if 'on_aspirin' in df.columns:
        categorical_results.extend(format_categorical_variable(df['on_aspirin'], 'Aspirin at discharge'))
    
    if 'p2y12_inhibitor' in df.columns:
        p2y12_labels = {
            0: 'None',
            1: 'Clopidogrel',
            2: 'Ticagrelor',
            3: 'Prasugrel',
            4: 'Other'
        }
        categorical_results.extend(
            format_categorical_variable(df['p2y12_inhibitor'], 'P2Y12 inhibitor', p2y12_labels)
        )
    
    if 'oral_anticoagulation' in df.columns:
        oral_anticoag_labels = {
            0: 'None',
            1: 'VKA',
            2: 'DOAC'
        }
        categorical_results.extend(
            format_categorical_variable(df['oral_anticoagulation'], 'Oral anticoagulation', oral_anticoag_labels)
        )
    
    categorical_df = pd.DataFrame(categorical_results)
    print(categorical_df.to_string(index=False))
    
    # Save to CSV
    procedural_table = pd.concat([continuous_df, categorical_df], ignore_index=True)
    procedural_table.to_csv('stats/results/baseline/procedural_characteristics.csv', index=False)
    
    return procedural_table

def create_lesion_evolution_table(df):
    """Create comprehensive lesion evolution metrics table"""
    print("\n" + "=" * 80)
    print("LESION EVOLUTION METRICS")
    print("=" * 80)
    print("-" * 80)
    
    metrics = []
    # Prepare subset excluding TLF if available
    df_no_tlf = None
    if 'tlf_fup_coro' in df.columns:
        df_no_tlf = df[df['tlf_fup_coro'] == 0]
    
    # MLD metrics
    mld_vars = [
        ('init_pre_mld', 'Baseline MLD (mm)'),
        ('init_post_mld', 'Post-procedural MLD (mm)'),
        ('fup_pre_mld', 'Follow-up MLD (mm)'),
        ('mld_acute_gain', 'MLD acute gain (mm)'),
        ('mld_late_lumen_change', 'MLD late lumen change (mm)'),
        ('mld_net_gain', 'MLD net gain (mm)'),
        ('mld_acute_recoil', 'MLD acute recoil (mm)'),
        ('mld_late_recoil', 'MLD late recoil (mm)')
    ]
    
    print("\nMLD Metrics:")
    for var, label in mld_vars:
        if var in df.columns:
            # All lesions
            metrics.append(format_continuous_variable(df[var], f"{label} [All]"))
            # Excluding TLF
            if df_no_tlf is not None and var in df_no_tlf.columns:
                metrics.append(format_continuous_variable(df_no_tlf[var], f"{label} [No TLF]"))
    

    
    # μFR metrics
    mufr_vars = [
        ('init_pre_mufr', 'Baseline μFR'),
        ('init_post_mufr', 'Post-procedural μFR'),
        ('fup_pre_mufr', 'Follow-up μFR'),
        ('mufr_acute_functional_gain', 'μFR acute functional gain'),
        ('mufr_late_functional_change', 'μFR late functional change'),
        ('mufr_net_functional_gain', 'μFR net functional gain')
    ]
    
    print("\nμFR Metrics:")
    for var, label in mufr_vars:
        if var in df.columns:
            # All lesions
            metrics.append(format_continuous_variable(df[var], f"{label} [All]"))
            # Excluding TLF
            if df_no_tlf is not None and var in df_no_tlf.columns:
                metrics.append(format_continuous_variable(df_no_tlf[var], f"{label} [No TLF]"))

    # Lesion length metrics
    length_vars = [
        ('init_pre_lesion_length', 'Baseline lesion length (mm)'),
        ('init_post_lesion_length', 'Post-procedural lesion length (mm)'),
        ('fup_pre_lesion_length', 'Follow-up lesion length (mm)'),
        ('fup_post_lesion_length', 'Follow-up post lesion length (mm)')
    ]

    print("\nLesion Length Metrics:")
    for var, label in length_vars:
        if var in df.columns:
            metrics.append(format_continuous_variable(df[var], f"{label} [All]"))
            if df_no_tlf is not None and var in df_no_tlf.columns:
                metrics.append(format_continuous_variable(df_no_tlf[var], f"{label} [No TLF]"))

    # Reference vessel diameter metrics
    ref_vars = [
        ('init_pre_ref_vessel_diam', 'Baseline reference diameter (mm)'),
        ('init_post_ref_vessel_diam', 'Post-procedural reference diameter (mm)'),
        ('fup_pre_ref_vessel_diam', 'Follow-up reference diameter (mm)'),
        ('fup_post_ref_vessel_diam', 'Follow-up post reference diameter (mm)')
    ]

    print("\nReference Vessel Diameter Metrics:")
    for var, label in ref_vars:
        if var in df.columns:
            metrics.append(format_continuous_variable(df[var], f"{label} [All]"))
            if df_no_tlf is not None and var in df_no_tlf.columns:
                metrics.append(format_continuous_variable(df_no_tlf[var], f"{label} [No TLF]"))
    
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df.to_string(index=False))
    
    # Save to CSV
    metrics_df.to_csv('stats/results/baseline/lesion_evolution_metrics.csv', index=False)
    
    # Clinical outcomes
    print("\n" + "-" * 80)
    print("CLINICAL OUTCOMES AT FOLLOW-UP")
    print("-" * 80)
    
    outcomes = []
    if 'tvf_fup_coro' in df.columns:
        outcomes.extend(format_categorical_variable(df['tvf_fup_coro'], 'Target vessel failure'))
    if 'tlf_fup_coro' in df.columns:
        outcomes.extend(format_categorical_variable(df['tlf_fup_coro'], 'Target lesion failure'))
    if 'is_residual_stenosis_at_fup' in df.columns:
        outcomes.extend(format_categorical_variable(df['is_residual_stenosis_at_fup'], 'Residual stenosis'))
    
    outcomes_df = pd.DataFrame(outcomes)
    print(outcomes_df.to_string(index=False))
    outcomes_df.to_csv('stats/results/baseline/clinical_outcomes.csv', index=False)
    
    return metrics_df, outcomes_df

    

def statistical_tests(df):
    """Perform statistical tests for changes between timepoints"""
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS FOR CHANGES BETWEEN TIMEPOINTS")
    print("=" * 80)
    
    # Test lumen gain vs loss groups
    if 'mld_late_lumen_change' in df.columns:
        df['lumen_gain_group'] = df['mld_late_lumen_change'] > 0
        
        print("\nComparison of Lumen Gain vs Loss Groups:")
        print("-" * 40)
        
        variables_to_compare = [
            ('init_post_mufr', 'Post-procedural μFR'),
            ('init_pre_mld', 'Baseline MLD'),
            ('init_pre_lesion_length', 'Baseline lesion length'),
            ('age_at_dcb', 'Age'),
            ('estimated_gfr', 'eGFR')
        ]
        
        for var, label in variables_to_compare:
            if var in df.columns:
                gain_group = df[df['lumen_gain_group'] == True][var].dropna()
                loss_group = df[df['lumen_gain_group'] == False][var].dropna()
                
                if len(gain_group) > 0 and len(loss_group) > 0:
                    t_stat, p_value = stats.ttest_ind(gain_group, loss_group)
                    
                    print(f"\n{label}:")
                    print(f"  Gain group: {gain_group.mean():.3f} ± {gain_group.std():.3f} (n={len(gain_group)})")
                    print(f"  Loss group: {loss_group.mean():.3f} ± {loss_group.std():.3f} (n={len(loss_group)})")
                    print(f"  p-value: {p_value:.4f} {'*' if p_value < 0.05 else 'ns'}")

def main():
    """Main execution function"""
    print("Starting Comprehensive DCB Lesion Evolution Analysis...")
    print("=" * 80)
    
    # Create directories for outputs
    create_directories()
    
    # Connect to database
    engine = connect_to_db()
    if engine is None:
        print("Failed to connect to database. Please check connection parameters.")
        return
    
    # Load joined data for baseline/procedural
    print("\nLoading data from camilla2 and denovo tables (joined)...")
    df_joined = load_data(engine)
    print(f"Loaded {len(df_joined)} lesions after inner join on date of birth")

    # Load metrics-only data directly from camilla2
    print("Loading metrics/outcomes directly from camilla2 (no join)...")
    df_metrics = load_metrics_data(engine)
    
    # Calculate derived variables
    df_joined = calculate_derived_variables(df_joined)
    df_metrics = calculate_derived_variables(df_metrics)
    
    # Create baseline characteristics table
    baseline_table = create_baseline_characteristics_table(df_joined)
    
    # Create procedural characteristics table
    procedural_table = create_procedural_characteristics_table(df_joined)
    
    # Create lesion evolution metrics table
    metrics_table, outcomes_table = create_lesion_evolution_table(df_metrics)
    
    # Skip plot generation per request
    
    # Perform statistical tests
    statistical_tests(df_metrics)
    
    # Save complete dataset
    # Save complete datasets
    df_joined.to_csv('stats/baseline/complete_dcb_analysis_joined.csv', index=False)
    df_metrics.to_csv('stats/baseline/complete_dcb_analysis_metrics.csv', index=False)
    print("\nComplete datasets saved to 'stats/baseline/complete_dcb_analysis_joined.csv' and 'stats/baseline/complete_dcb_analysis_metrics.csv'")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Joined set size: {len(df_joined)} | Metrics set size: {len(df_metrics)}")
    print(f"Follow-up duration (metrics set): {df_metrics['months_to_followup'].mean():.1f} ± {df_metrics['months_to_followup'].std():.1f} months")
    
    if 'mld_late_lumen_change' in df_metrics.columns:
        gain_pct = (df_metrics['mld_late_lumen_change'] > 0).mean() * 100
        print(f"Lesions with lumen gain: {gain_pct:.1f}%")
    
    if 'tlf_fup_coro' in df_metrics.columns:
        tlf_rate = df_metrics['tlf_fup_coro'].mean() * 100
        print(f"Overall TLF rate: {tlf_rate:.1f}%")
    
    print("\nAll results saved to 'stats/baseline/' directory")
    print("\nAnalysis complete!")
    
    return df_metrics

if __name__ == "__main__":
    # Run the analysis
    df_results = main()