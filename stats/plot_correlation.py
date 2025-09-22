#!/usr/bin/env python3
"""
Scatter Plot Analysis: Visualizing Relationships Between Key Predictors and Outcomes
Creates beautiful scatter plots with regression lines for each predictor-outcome pair
Enhanced with modern design elements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

# Enhanced style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.1
plt.rcParams['grid.linewidth'] = 0.5

# Define modern color palette
COLORS = {
    'primary_blue': '#2E86AB',
    'primary_purple': '#7209B7', 
    'success_green': '#06B6D4',
    'danger_red': '#F72585',
    'neutral_gray': '#6B7280',
    'light_gray': '#F3F4F6',
    'dark_gray': '#374151',
    'accent_orange': '#FF6B6B',
    'accent_teal': '#4ECDC4',
    'light_blue': '#A5C9EA',
    'light_purple': '#C77DFF'
}

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'dcb',
    'user': 'doriangarin',
    'password': '96349130dG!',
    'port': 5432
}

# Define the key predictors and outcomes
KEY_PREDICTORS = [
    'dcb_inflation_max_time',
    'dcb_max_pressure',
    'dcb_diam_to_vessel',
    'predilatation_diam_to_vessel'
]

OUTCOMES = [
    'mld_late_lumen_change',
    'mufr_late_functional_change'
]

# Prettier names for plotting
PREDICTOR_LABELS = {
    'dcb_inflation_max_time': 'DCB inflation time (sec)',
    'dcb_max_pressure': 'DCB maximum pressure (atm)',
    'dcb_diam_to_vessel': 'DCB diameter to vessel ratio',
    'predilatation_diam_to_vessel': 'Predilatation balloon diameter to vessel ratio'
}

OUTCOME_LABELS = {
    'mld_late_lumen_change': 'Late lumen gain (mm)',
    'mufr_late_functional_change': 'Late functional gain'
}

def connect_to_db():
    """Create database connection"""
    try:
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def load_data(engine):
    """Load relevant data from database"""
    query = """
    SELECT 
        c.dcb_inflation_max_time,
        c.mld_late_lumen_change,
        c.mufr_late_functional_change,
        c.dcb_diam_to_vessel,
        c.dcb_lenght_to_vessel,
        c.predilatation_diam_to_vessel,
        d.dcb_1_inflation_pressure as dcb_max_pressure
    FROM camilla2 c
    INNER JOIN denovo d ON c.dob = d.birth_date
    """
    
    df = pd.read_sql(query, engine)
    return df

def create_scatter_plot(df, predictor, outcome, ax):
    """
    Create a single scatter plot with enhanced design and regression line
    """
    # Clean data - remove NaN values
    clean_df = df[[predictor, outcome]].dropna()
    
    if len(clean_df) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color=COLORS['neutral_gray'])
        ax.set_xticks([])
        ax.set_yticks([])
        return
    
    x = clean_df[predictor]
    y = clean_df[outcome]
    
    # Calculate statistics first
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Determine plot color based on outcome
    if 'mld' in outcome.lower():
        main_color = COLORS['primary_blue']
        light_color = COLORS['light_blue']
    elif 'mufr' in outcome.lower() or 'μfr' in outcome.lower():
        main_color = COLORS['primary_purple']
        light_color = COLORS['light_purple']
    else:
        main_color = COLORS['accent_teal']
        light_color = COLORS['success_green']
    
    # Add subtle gradient background for positive/negative regions
    y_min, y_max = ax.get_ylim() if ax.get_ylim()[0] != 0 else (y.min()*1.1, y.max()*1.1)
    
    # Only add background if outcome can be negative
    has_negative = y.min() < 0
    has_positive = y.max() > 0
    
    if has_negative and has_positive:
        ax.axhspan(0, y_max*1.5, alpha=0.02, color=COLORS['success_green'], zorder=0)
        ax.axhspan(y_min*1.5, 0, alpha=0.02, color=COLORS['danger_red'], zorder=0)
        
        # Add zero line
        ax.axhline(y=0, color=COLORS['dark_gray'], linestyle='-', linewidth=1.5, alpha=0.3, zorder=1)
    
    # Track which types of points we have for legend
    has_pos_points = False
    has_neg_points = False
    
    # Create scatter plot with colors based on positive/negative values
    for i in range(len(x)):
        # Color based on value
        if has_negative and has_positive:
            if y.iloc[i] > 0:
                point_color = COLORS['success_green']
                has_pos_points = True
            else:
                point_color = COLORS['danger_red']
                has_neg_points = True
        else:
            # All same sign - use main color
            point_color = main_color
        
        ax.scatter(x.iloc[i], y.iloc[i], 
                  alpha=0.7, s=80, 
                  color=point_color,
                  edgecolors='white', linewidth=1.5, zorder=3)
    
    # Add regression line with confidence interval
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = slope * x_smooth + intercept
    
    # Calculate confidence interval
    predict_y = slope * x + intercept
    residuals = y - predict_y
    se = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
    
    # 95% confidence interval
    t_val = stats.t.ppf(0.975, len(x) - 2)
    margin = t_val * se * np.sqrt(1/len(x) + (x_smooth - x.mean())**2 / np.sum((x - x.mean())**2))
    
    # Plot confidence interval
    ax.fill_between(x_smooth, y_smooth - margin, y_smooth + margin, 
                    alpha=0.15, color=main_color, zorder=2)
    
    # Plot regression line
    ax.plot(x_smooth, y_smooth, color=main_color, linewidth=3, 
           zorder=4, alpha=0.9, label='Regression line')
    
    # Enhanced axis formatting
    ax.set_xlabel(PREDICTOR_LABELS.get(predictor, predictor), 
                 fontsize=12, fontweight='medium')
    ax.set_ylabel(OUTCOME_LABELS.get(outcome, outcome), 
                 fontsize=12, fontweight='medium')
    
    # Enhanced grid
    ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spine formatting
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color(COLORS['dark_gray'])
    ax.spines['bottom'].set_color(COLORS['dark_gray'])
    
    # Determine significance level
    if p_value < 0.001:
        sig_text = "***"
        p_text = "p < 0.001"
    elif p_value < 0.01:
        sig_text = "**"
        p_text = f"p = {p_value:.3f}"
    elif p_value < 0.05:
        sig_text = "*"
        p_text = f"p = {p_value:.3f}"
    else:
        sig_text = "ns"
        p_text = f"p = {p_value:.2f}"
    
    # Add enhanced statistics box
    stats_text = (
        f"r² = {r_value**2:.3f}\n"
        f"{p_text}"
    )
    
    # Enhanced text box styling
    bbox_props = dict(boxstyle='round,pad=0.5', 
                     facecolor='white', 
                     edgecolor=main_color,
                     linewidth=1.5,
                     alpha=0.95)
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=bbox_props)

    
    # Add trend indicator for significant relationships
    if p_value < 0.05:
        if slope > 0:
            trend_symbol = "↑"
            trend_position = 0.98
        else:
            trend_symbol = "↓"
            trend_position = 0.02
  
    # Add legend for point colors
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D
    legend_elements = []
    
    # Add regression line to legend
    legend_elements.append(Line2D([0], [0], color=main_color, linewidth=3, 
                                  alpha=0.9, label='Regression line'))
    
    # Add confidence interval to legend
    legend_elements.append(Line2D([0], [0], color=main_color, linewidth=8, 
                                  alpha=0.15, label='95% CI'))
    
    # Add point types based on what's in the plot
    if has_negative and has_positive:
        # Mixed positive/negative values
        if has_pos_points:
            legend_elements.append(Circle((0, 0), 1, facecolor=COLORS['success_green'], 
                                         edgecolor='white', linewidth=1.5, alpha=0.7,
                                         label='Gain (positive)'))
        if has_neg_points:
            legend_elements.append(Circle((0, 0), 1, facecolor=COLORS['danger_red'],
                                         edgecolor='white', linewidth=1.5, alpha=0.7,
                                         label='Loss (negative)'))
    else:
        # All same sign - show main color
        if has_positive:
            label_text = 'Data points (positive)'
        elif has_negative:
            label_text = 'Data points (negative)'
        else:
            label_text = 'Data points'
        
        legend_elements.append(Circle((0, 0), 1, facecolor=main_color,
                                     edgecolor='white', linewidth=1.5, alpha=0.7,
                                     label=label_text))
    
    # Position legend based on data distribution
    # Check quadrants to find best position
    x_median = x.median()
    y_median = y.median()
    
    # Count points in each quadrant
    top_right = ((x > x_median) & (y > y_median)).sum()
    top_left = ((x < x_median) & (y > y_median)).sum()
    bottom_right = ((x > x_median) & (y < y_median)).sum()
    bottom_left = ((x < x_median) & (y < y_median)).sum()
    
    # Find quadrant with fewest points
    quadrant_counts = {
        'upper right': top_right,
        'upper left': top_left,
        'lower right': bottom_right,
        'lower left': bottom_left
    }
    best_position = min(quadrant_counts, key=quadrant_counts.get)
    
    # Add legend to plot
    legend = ax.legend(handles=legend_elements, 
                      loc=best_position,
                      frameon=True,
                      fancybox=True,
                      shadow=False,
                      framealpha=0.92,
                      edgecolor=main_color,
                      fontsize=9,
                      title='Legend',
                      title_fontsize=10)
    
    # Style the legend
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_linewidth(1.5)
    
    # Add legend for point colors if outcome can be negative
    if y.min() < 0:
        # Create legend elements
        from matplotlib.patches import Patch
        legend_elements = [
            plt.scatter([], [], color=COLORS['success_green'], s=80, 
                       edgecolors='white', linewidth=1.5, alpha=0.7, label='Positive change'),
            plt.scatter([], [], color=COLORS['danger_red'], s=80, 
                       edgecolors='white', linewidth=1.5, alpha=0.7, label='Negative change')
        ]
        
        # Add legend in bottom right
        legend = ax.legend(handles=legend_elements, loc='lower right',
                          frameon=True, fancybox=True, shadow=False,
                          framealpha=0.95, edgecolor=main_color,
                          fontsize=9, title='Point types', title_fontsize=10)
        legend.get_frame().set_linewidth(1.5)

def create_individual_plots(df):
    """Create individual plots for each predictor-outcome pair with enhanced design"""
    import os
    os.makedirs('plots/individual', exist_ok=True)
    
    for predictor in KEY_PREDICTORS:
        if predictor not in df.columns:
            print(f"  ⚠️  Warning: {predictor} not found in data")
            continue
            
        for outcome in OUTCOMES:
            if outcome not in df.columns:
                print(f"  ⚠️  Warning: {outcome} not found in data")
                continue
            
            # Create figure with enhanced design
            fig, ax = plt.subplots(figsize=(9, 7))
            fig.patch.set_facecolor('white')
            
            create_scatter_plot(df, predictor, outcome, ax)
            
            # Enhanced title with color coding based on outcome
            if 'mld' in outcome.lower():
                title_color = COLORS['primary_blue']
            elif 'mufr' in outcome.lower() or 'μfr' in outcome.lower():
                title_color = COLORS['primary_purple']
            else:
                title_color = COLORS['dark_gray']
   
        

            
            plt.tight_layout()
            
            # Save with descriptive filename
            filename = f'plots/individual/{predictor}_vs_{outcome}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"  ✓ Saved: {filename}")

def create_combined_plot(df):
    """Create a combined plot showing all predictor-outcome relationships with enhanced design"""
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Count valid predictors and outcomes
    valid_predictors = [p for p in KEY_PREDICTORS if p in df.columns]
    valid_outcomes = [o for o in OUTCOMES if o in df.columns]
    
    n_predictors = len(valid_predictors)
    n_outcomes = len(valid_outcomes)
    
    if n_predictors == 0 or n_outcomes == 0:
        print("No valid predictors or outcomes found")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_outcomes, n_predictors, 
                             figsize=(5.5*n_predictors, 5*n_outcomes))
    fig.patch.set_facecolor('white')
    
    # Handle single row/column case
    if n_outcomes == 1:
        axes = axes.reshape(1, -1)
    if n_predictors == 1:
        axes = axes.reshape(-1, 1)
    
    for i, outcome in enumerate(valid_outcomes):
        for j, predictor in enumerate(valid_predictors):
            ax = axes[i, j]
            create_scatter_plot(df, predictor, outcome, ax)
            
            # Add subplot title only for top row
            
    
    
    
    plt.tight_layout()
    
    # Save combined plot
    filename = 'plots/all_predictors_outcomes_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved combined plot: {filename}")

def create_correlation_heatmap(df):
    """Create a correlation heatmap for predictors and outcomes with enhanced design"""
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Select variables
    variables = []
    var_labels = {}
    
    for pred in KEY_PREDICTORS:
        if pred in df.columns:
            variables.append(pred)
            var_labels[pred] = PREDICTOR_LABELS.get(pred, pred)
    
    for outcome in OUTCOMES:
        if outcome in df.columns:
            variables.append(outcome)
            var_labels[outcome] = OUTCOME_LABELS.get(outcome, outcome)
    
    if len(variables) < 2:
        print("Not enough variables for correlation heatmap")
        return
    
    # Calculate correlations
    corr_df = df[variables].corr()
    
    # Create figure with enhanced design
    plt.figure(figsize=(11, 9))
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
    
    # Create custom labels
    labels = [var_labels[var] for var in corr_df.columns]
    
    # Custom colormap
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    # Draw heatmap with enhanced styling
    ax = sns.heatmap(corr_df, mask=mask, annot=True, fmt='.3f', 
                    cmap=cmap, center=0, square=True,
                    linewidths=2, linecolor='white',
                    cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                    vmin=-1, vmax=1,
                    annot_kws={'fontsize': 10, 'fontweight': 'medium'})
    
    # Set labels with enhanced formatting
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(labels, rotation=0, fontsize=11)
    
    # Enhanced title
    plt.title('Correlation Matrix: Predictors and Outcomes', 
             fontsize=16, fontweight='bold', pad=20)
    
    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor(COLORS['dark_gray'])
    
    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Correlation Coefficient', fontsize=11, fontweight='medium')
    
    plt.tight_layout()
    
    # Save
    filename = 'plots/correlation_heatmap.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"Saved correlation heatmap: {filename}")
    
    # Print correlation values with enhanced formatting
    print("\n" + "═"*60)
    print(" "*20 + "CORRELATION ANALYSIS")
    print("═"*60)
    
    for outcome in OUTCOMES:
        if outcome in corr_df.columns:
            print(f"\n📊 {OUTCOME_LABELS.get(outcome, outcome)}:")
            print("─"*60)
            
            # Sort correlations by absolute value
            outcome_corrs = corr_df[outcome].drop(outcome).abs().sort_values(ascending=False)
            
            for pred in outcome_corrs.index:
                if pred in KEY_PREDICTORS:
                    corr_val = corr_df.loc[pred, outcome]
                    
                    # Determine strength
                    abs_corr = abs(corr_val)
                    if abs_corr > 0.7:
                        strength = "Strong"
                        symbol = "●●●"
                    elif abs_corr > 0.5:
                        strength = "Moderate"
                        symbol = "●●○"
                    elif abs_corr > 0.3:
                        strength = "Weak"
                        symbol = "●○○"
                    else:
                        strength = "Negligible"
                        symbol = "○○○"
                  

def perform_logistic_regression_analysis(df):
    """Perform logistic regression analysis: positive vs negative late changes"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, roc_curve
    from sklearn.metrics import confusion_matrix
    import os
    
    os.makedirs('plots/logistic', exist_ok=True)
    
    print("\n" + "═"*60)
    print(" "*15 + "LOGISTIC REGRESSION ANALYSIS")
    print(" "*10 + "Predicting Positive vs Negative Changes")
    print("═"*60)
    
    results_summary = []
    
    for outcome in OUTCOMES:
        if outcome not in df.columns:
            continue
        
        print(f"\n📊 Analyzing: {OUTCOME_LABELS.get(outcome, outcome)}")
        print("─"*60)
        
        # Prepare data
        analysis_df = df.copy()
        
        # Create binary outcome (1 = positive change, 0 = negative/no change)
        analysis_df['outcome_binary'] = (analysis_df[outcome] > 0).astype(int)
        
        # Select predictors and remove missing values
        predictors = [p for p in KEY_PREDICTORS if p in analysis_df.columns]
        
        # Create complete dataset
        complete_data = analysis_df[predictors + ['outcome_binary']].dropna()
        
        if len(complete_data) < 20:
            print(f"  ⚠️  Insufficient data (n={len(complete_data)})")
            continue
        
        # Prepare features and target
        X = complete_data[predictors]
        y = complete_data['outcome_binary']
        
        # Count positive and negative cases
        n_positive = y.sum()
        n_negative = len(y) - n_positive
        pct_positive = (n_positive / len(y)) * 100
        
        print(f"\n  Sample Statistics:")
        print(f"    • Total samples: {len(y)}")
        print(f"    • Positive changes: {n_positive} ({pct_positive:.1f}%)")
        print(f"    • Negative changes: {n_negative} ({100-pct_positive:.1f}%)")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit logistic regression
        log_reg = LogisticRegression(random_state=42, max_iter=1000)
        log_reg.fit(X_scaled, y)
        
        # Get predictions and probabilities
        y_pred = log_reg.predict(X_scaled)
        y_prob = log_reg.predict_proba(X_scaled)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y, y_prob)
        
        # Get feature importance (coefficients)
        coefficients = pd.DataFrame({
            'Predictor': predictors,
            'Coefficient': log_reg.coef_[0],
            'Abs_Coefficient': np.abs(log_reg.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(f"\n  Model Performance:")
        print(f"    • AUC-ROC: {auc:.3f}")
        
        # Print top predictors
        print(f"\n  Top Predictors (by coefficient magnitude):")
        
            
        
        # Store results
        results_summary.append({
            'Outcome': OUTCOME_LABELS.get(outcome, outcome),
            'N': len(y),
            '% Positive': pct_positive,
            'AUC': auc,
            'Top Predictor': coefficients.iloc[0]['Predictor']
        })
        
        # Create visualization: ROC curve and feature importance
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor('white')
        
        # Determine colors based on outcome
        if 'mld' in outcome.lower():
            main_color = COLORS['primary_blue']
            light_color = COLORS['light_blue']
        else:
            main_color = COLORS['primary_purple']
            light_color = COLORS['light_purple']
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_prob)
        ax1.plot(fpr, tpr, color=main_color, linewidth=3,
                label=f'ROC curve (AUC = {auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5)
        ax1.fill_between(fpr, tpr, alpha=0.15, color=main_color)
        
        ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='medium')
        ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='medium')
        ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=11, frameon=True, 
                  fancybox=True, framealpha=0.95, edgecolor=main_color)
        ax1.grid(True, alpha=0.1)
        ax1.spines['left'].set_linewidth(1.2)
        ax1.spines['bottom'].set_linewidth(1.2)
        
        # Feature Importance Plot
        top_features = coefficients.head(10)
        y_pos = np.arange(len(top_features))
        
        # Color bars based on coefficient sign
        colors = [COLORS['success_green'] if c > 0 else COLORS['danger_red'] 
                 for c in top_features['Coefficient'].values]
        
        bars = ax2.barh(y_pos, top_features['Coefficient'].values, color=colors, 
                       alpha=0.7, edgecolor='white', linewidth=2)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([PREDICTOR_LABELS.get(p, p) 
                            for p in top_features['Predictor'].values], fontsize=10)
        ax2.set_xlabel('Coefficient', fontsize=12, fontweight='medium')
        ax2.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color=COLORS['dark_gray'], linestyle='-', 
                   linewidth=1.5, alpha=0.3)
        ax2.grid(True, alpha=0.1, axis='x')
        ax2.spines['left'].set_linewidth(1.2)
        ax2.spines['bottom'].set_linewidth(1.2)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, top_features['Coefficient'].values)):
            ax2.text(val + (0.05 if val > 0 else -0.05), bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}', ha='left' if val > 0 else 'right',
                    va='center', fontsize=9, fontweight='medium')
        
        # Main title for the figure
        outcome_name = OUTCOME_LABELS.get(outcome, outcome)
        fig.suptitle(f'Logistic Regression: Predicting Positive {outcome_name}',
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'plots/logistic/logistic_{outcome}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"\n  ✓ Saved plot: {filename}")
        
    # Create summary table
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df.to_csv('plots/logistic/logistic_summary.csv', index=False)
        
        print("\n" + "═"*60)
        print(" "*20 + "SUMMARY TABLE")
        print("═"*60)
        print(summary_df.to_string(index=False))
        print("\n✓ Summary saved to: plots/logistic/logistic_summary.csv")
    
    return results_summary

def generate_summary_statistics(df):
    """Generate summary statistics for all variables with enhanced formatting"""
    import os
    os.makedirs('plots', exist_ok=True)
    
    summary = []
    
    # Add predictors
    for pred in KEY_PREDICTORS:
        if pred in df.columns:
            data = df[pred].dropna()
            summary.append({
                'Variable': PREDICTOR_LABELS.get(pred, pred),
                'Type': 'Predictor',
                'N': len(data),
                'Mean': data.mean(),
                'SD': data.std(),
                'Median': data.median(),
                'Q1': data.quantile(0.25),
                'Q3': data.quantile(0.75),
                'Min': data.min(),
                'Max': data.max()
            })
    
    # Add outcomes
    for outcome in OUTCOMES:
        if outcome in df.columns:
            data = df[outcome].dropna()
            
            # Calculate percentage with positive change
            pct_positive = (data > 0).mean() * 100
            
            summary.append({
                'Variable': OUTCOME_LABELS.get(outcome, outcome),
                'Type': 'Outcome',
                'N': len(data),
                'Mean': data.mean(),
                'SD': data.std(),
                'Median': data.median(),
                'Q1': data.quantile(0.25),
                'Q3': data.quantile(0.75),
                'Min': data.min(),
                'Max': data.max(),
                '% Positive': pct_positive
            })
    
    summary_df = pd.DataFrame(summary)
    
    # Save to CSV
    summary_df.to_csv('plots/summary_statistics.csv', index=False)
    
    # Print formatted summary with enhanced design
    print("\n" + "═"*80)
    print(" "*30 + "SUMMARY STATISTICS")
    print("═"*80)
    
    # Separate by type
    print("\n📊 PREDICTORS:")
    print("─"*80)
    for _, row in summary_df[summary_df['Type'] == 'Predictor'].iterrows():
        print(f"\n{row['Variable']}:")
        print(f"  • Sample size: n = {row['N']}")
        print(f"  • Mean ± SD: {row['Mean']:.3f} ± {row['SD']:.3f}")
        print(f"  • Median [IQR]: {row['Median']:.3f} [{row['Q1']:.3f}, {row['Q3']:.3f}]")
        print(f"  • Range: {row['Min']:.3f} to {row['Max']:.3f}")
    
    print("\n📈 OUTCOMES:")
    print("─"*80)
    for _, row in summary_df[summary_df['Type'] == 'Outcome'].iterrows():
        print(f"\n{row['Variable']}:")
        print(f"  • Sample size: n = {row['N']}")
        print(f"  • Mean ± SD: {row['Mean']:.3f} ± {row['SD']:.3f}")
        print(f"  • Median [IQR]: {row['Median']:.3f} [{row['Q1']:.3f}, {row['Q3']:.3f}]")
        print(f"  • Range: {row['Min']:.3f} to {row['Max']:.3f}")
       

def main():
    """Main execution function with enhanced formatting"""
    print("\n" + "═"*80)
    print(" "*20 + "SCATTER PLOT ANALYSIS")
    print(" "*15 + "Key Predictors vs Late Outcomes")
    print("═"*80)
    
    # Connect to database
    print("\n🔄 Connecting to database...")
    engine = connect_to_db()
    if engine is None:
        print("❌ Failed to connect to database")
        return
    
    # Load data
    print("📊 Loading data...")
    df = load_data(engine)
    print(f"✅ Loaded {len(df)} observations")
    
    # Generate summary statistics
    print("\n📈 Analyzing variables...")
    generate_summary_statistics(df)
    
    # Create plots
    print("\n🎨 Generating visualizations...")
    
    # 1. Individual plots for each pair
    print("\n  1. Creating individual scatter plots...")
    create_individual_plots(df)
    
    # 2. Combined plot
    print("\n  2. Creating combined overview plot...")
    create_combined_plot(df)
    
    # 3. Correlation heatmap
    print("\n  3. Creating correlation heatmap...")
    create_correlation_heatmap(df)
    
    # 4. Logistic regression analysis
    print("\n  4. Performing logistic regression analysis...")
    logistic_results = perform_logistic_regression_analysis(df)
    
    print("\n" + "═"*80)
    print(" "*25 + "ANALYSIS COMPLETE")
    print("═"*80)
    
    print("\n✅ Results saved to:")
    print("  📁 Individual plots: plots/individual/")
    print("  📊 Combined plot: plots/all_predictors_outcomes_combined.png")
    print("  🔥 Correlation heatmap: plots/correlation_heatmap.png")
    print("  📈 Logistic regression: plots/logistic/")
    print("  📄 Summary statistics: plots/summary_statistics.csv")
    
    print("\n💡 Key insights:")
    print("  • Scatter plots show relationships between procedural parameters and outcomes")
    print("  • Regression lines indicate direction and strength of associations")
    print("  • Statistical significance marked with asterisks (*, **, ***)")
    print("  • Correlation matrix reveals multicollinearity patterns")
    print("  • Logistic regression identifies predictors of positive vs negative changes")
    print("  • Color-coded points: green=positive, red=negative, gray=outliers")
    
    return df

if __name__ == "__main__":
    df = main()