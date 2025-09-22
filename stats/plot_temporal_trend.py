import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
import os
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
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.15
plt.rcParams['grid.linewidth'] = 0.8

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
    'accent_teal': '#4ECDC4'
}

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'dcb',
    'user': 'doriangarin',
    'password': '96349130dG!',
    'port': 5432
}

def create_directories():
    """Create directories for saving results"""
    os.makedirs('stats/plots/temporal', exist_ok=True)
    os.makedirs('stats/results/temporal', exist_ok=True)

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
    """Load all necessary data"""
    query = """
    SELECT 
        -- Temporal variables
        date_dcb,
        date_fup_coro,
        
        -- For paired analysis
        init_post_mld,
        fup_post_mld,
        init_post_mufr,
        fup_post_mufr,
        
        -- For time regression
        mld_late_lumen_change,
        mufr_late_functional_change
        
    FROM camilla2
    WHERE 1=1


    """
    
    df = pd.read_sql(query, engine)
    
    # Convert dates and calculate follow-up time
    df['date_dcb'] = pd.to_datetime(df['date_dcb'])
    df['date_fup_coro'] = pd.to_datetime(df['date_fup_coro'])
    df['months_to_followup'] = (df['date_fup_coro'] - df['date_dcb']).dt.days / 30.44
    
    return df

def plot_combined_paired_analysis(df):
    """Create combined paired analysis plot for MLD and μFR (init post → fup post)"""
    
    # Get paired data for both metrics
    mld_paired = df[['init_post_mld', 'fup_post_mld']].dropna()
    mufr_paired = df[['init_post_mufr', 'fup_post_mufr']].dropna()
    
    if len(mld_paired) < 3 or len(mufr_paired) < 3:
        print("Insufficient data for paired analysis")
        return None
    
    # Calculate statistics
    mld_changes = mld_paired['fup_post_mld'] - mld_paired['init_post_mld']
    mufr_changes = mufr_paired['fup_post_mufr'] - mufr_paired['init_post_mufr']
    
    mld_t_stat, mld_p = stats.ttest_rel(mld_paired['fup_post_mld'], mld_paired['init_post_mld'])
    mufr_t_stat, mufr_p = stats.ttest_rel(mufr_paired['fup_post_mufr'], mufr_paired['init_post_mufr'])
    
    # Create figure with enhanced styling
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.patch.set_facecolor('white')
    
    # MLD subplot
    mld_mean_post = mld_paired['init_post_mld'].mean()
    mld_mean_fup = mld_paired['fup_post_mld'].mean()
    mld_std_post = mld_paired['init_post_mld'].std()
    mld_std_fup = mld_paired['fup_post_mld'].std()
    
    # Plot individual lines with transparency for MLD
    for i in range(len(mld_paired)):
        ax1.plot([0, 1], 
                [mld_paired.iloc[i]['init_post_mld'], mld_paired.iloc[i]['fup_post_mld']], 
                color=COLORS['neutral_gray'], alpha=0.15, linewidth=0.8, zorder=1)
    
    # Plot mean line with error bars for MLD
    ax1.errorbar([0, 1], [mld_mean_post, mld_mean_fup], 
                yerr=[mld_std_post/np.sqrt(len(mld_paired)), mld_std_fup/np.sqrt(len(mld_paired))],
                color=COLORS['primary_blue'], linewidth=3, marker='o', markersize=14,
                capsize=6, capthick=2.5, label='Mean ± SEM', zorder=3,
                markeredgecolor='white', markeredgewidth=2)
    
    # Formatting MLD
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Post-procedural', 'Follow-up'], fontsize=13, fontweight='medium')
    ax1.set_ylabel('MLD (mm)', fontsize=14, fontweight='medium')
    ax1.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)
    ax1.set_xlim(-0.25, 1.25)
    ax1.spines['left'].set_linewidth(1.2)
    ax1.spines['bottom'].set_linewidth(1.2)
    
    # Add shaded region for change in MLD
    if mld_changes.mean() > 0:
        ax1.axhspan(mld_mean_post, mld_mean_fup, alpha=0.05, color=COLORS['success_green'], zorder=0)
    else:
        ax1.axhspan(mld_mean_fup, mld_mean_post, alpha=0.05, color=COLORS['danger_red'], zorder=0)
    
    # Add statistics text for MLD with enhanced styling
    mld_relative_change = (mld_changes.mean() / mld_paired['init_post_mld'].mean()) * 100
    
    # Determine significance level
    if mld_p < 0.001:
        sig_text = "***"
    elif mld_p < 0.01:
        sig_text = "**"
    elif mld_p < 0.05:
        sig_text = "*"
    else:
        sig_text = ""
    
    mld_stats_text = (
        f"Δ = {mld_changes.mean():.3f} mm\n"
        f"Relative Δ = {mld_relative_change:.1f}%\n"
        f"p = {mld_p:.3f} {sig_text}"
    )
    
    ax1.text(0.03, 0.97, mld_stats_text, transform=ax1.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95, 
                     edgecolor=COLORS['primary_blue'], linewidth=1.5))
    
    # μFR subplot
    mufr_mean_post = mufr_paired['init_post_mufr'].mean()
    mufr_mean_fup = mufr_paired['fup_post_mufr'].mean()
    mufr_std_post = mufr_paired['init_post_mufr'].std()
    mufr_std_fup = mufr_paired['fup_post_mufr'].std()
    
    # Plot individual lines with transparency for μFR
    for i in range(len(mufr_paired)):
        ax2.plot([0, 1], 
                [mufr_paired.iloc[i]['init_post_mufr'], mufr_paired.iloc[i]['fup_post_mufr']], 
                color=COLORS['neutral_gray'], alpha=0.15, linewidth=0.8, zorder=1)
    
    # Plot mean line with error bars for μFR
    ax2.errorbar([0, 1], [mufr_mean_post, mufr_mean_fup], 
                yerr=[mufr_std_post/np.sqrt(len(mufr_paired)), mufr_std_fup/np.sqrt(len(mufr_paired))],
                color=COLORS['primary_purple'], linewidth=3, marker='o', markersize=14,
                capsize=6, capthick=2.5, label='Mean ± SEM', zorder=3,
                markeredgecolor='white', markeredgewidth=2)
    
    # Formatting μFR
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Post-procedural', 'Follow-up'], fontsize=13, fontweight='medium')
    ax2.set_ylabel('μFR', fontsize=14, fontweight='medium')
    ax2.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)
    ax2.set_xlim(-0.25, 1.25)
    ax2.set_ylim(70, 100)  # Fixed y-axis for μFR
    ax2.set_yticks([70, 80, 90, 100])
    ax2.spines['left'].set_linewidth(1.2)
    ax2.spines['bottom'].set_linewidth(1.2)
    
    
    
    # Add shaded region for change in μFR (baseline at 80)
    if mufr_mean_fup > 80 and mufr_mean_post < 80:
        # Crossed above baseline
        ax2.axhspan(mufr_mean_post, mufr_mean_fup, alpha=0.05, color=COLORS['success_green'], zorder=0)
    elif mufr_mean_fup < 80 and mufr_mean_post > 80:
        # Crossed below baseline
        ax2.axhspan(mufr_mean_fup, mufr_mean_post, alpha=0.05, color=COLORS['danger_red'], zorder=0)
    elif mufr_changes.mean() > 0:
        # Both above or below but improving
        ax2.axhspan(mufr_mean_post, mufr_mean_fup, alpha=0.05, color=COLORS['success_green'], zorder=0)
    else:
        # Both above or below but declining
        ax2.axhspan(mufr_mean_fup, mufr_mean_post, alpha=0.05, color=COLORS['danger_red'], zorder=0)
    
    # Add statistics text for μFR
    mufr_relative_change = (mufr_changes.mean() / mufr_paired['init_post_mufr'].mean()) * 100
    
    # Determine significance level
    if mufr_p < 0.001:
        sig_text = "***"
    elif mufr_p < 0.01:
        sig_text = "**"
    elif mufr_p < 0.05:
        sig_text = "*"
    else:
        sig_text = "ns"
    
    mufr_stats_text = (
        f"Δ = {mufr_changes.mean():.2f}\n"
        f"Relative Δ = {mufr_relative_change:.1f}%\n"
        f"p = {mufr_p:.3f} {sig_text}"
    )
    
    ax2.text(0.03, 0.97, mufr_stats_text, transform=ax2.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95,
                     edgecolor=COLORS['primary_purple'], linewidth=1.5))
    

    
    plt.tight_layout()
    
    # Save
    filename = 'stats/plots/temporal/combined_paired_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Saved: {filename}")
    
    return {
        'mld': {'n': len(mld_paired), 'mean_change': mld_changes.mean(), 'p_value': mld_p},
        'mufr': {'n': len(mufr_paired), 'mean_change': mufr_changes.mean(), 'p_value': mufr_p}
    }

def plot_time_regression(df, metric='mld'):
    """Create regression plot of late changes over time with enhanced design"""
    
    config = {
        'mld': {
            'column': 'mld_late_lumen_change',
            'ylabel': 'Late lumen gain (mm)',
            'title': 'Minimal Lumen Diameter Change Over Time',
            'color': COLORS['primary_blue'],
            'color_light': '#A5C9EA'
        },
        'mufr': {
            'column': 'mufr_late_functional_change',
            'ylabel': 'Late functional gain',
            'title': 'Microvascular Flow Reserve Change Over Time',
            'color': COLORS['primary_purple'],
            'color_light': '#C77DFF'
        }
    }
    
    params = config[metric]
    
    # Get data
    data = df[['months_to_followup', params['column']]].dropna()
    
    if len(data) < 3:
        print(f"Insufficient data for {metric} regression")
        return None
    
    x = data['months_to_followup'].values.reshape(-1, 1)
    y = data[params['column']].values
    
    # Fit linear regression
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    # Calculate p-value and confidence interval
    n = len(x)
    t_stat = model.coef_[0] / (np.sqrt(np.sum((y - y_pred)**2) / (n-2)) / np.sqrt(np.sum((x - x.mean())**2)))
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))
    
    # Calculate R²
    r_squared = model.score(x, y)
    
    # Calculate percentiles for outlier detection (different from axis limits)
    percentile_low = np.percentile(y, 5)
    percentile_high = np.percentile(y, 95)
    
    # Create plot with enhanced design
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('white')
    
    # Set y-axis limits
    if metric == 'mufr':
        # For μFR CHANGES, auto-scale based on data
        # Calculate regression line range at plot boundaries
        x_plot_min, x_plot_max = x.min(), x.max()
        y_reg_min = model.predict([[x_plot_min]])[0]
        y_reg_max = model.predict([[x_plot_max]])[0]
        y_reg_center = (y_reg_min + y_reg_max) / 2
        
        # Set y-axis limits centered on regression line
        y_range = percentile_high - percentile_low
        y_padding = y_range * 0.6
        
        y_limit_min = min(percentile_low, y_reg_center - y_padding)
        y_limit_max = max(percentile_high, y_reg_center + y_padding)
        
        # Ensure zero is visible if regression line crosses it
        if y_reg_min * y_reg_max < 0 or abs(y_reg_center) < y_padding:
            y_limit_min = min(y_limit_min, -y_padding * 0.5)
            y_limit_max = max(y_limit_max, y_padding * 0.5)
    else:
        # For MLD, same auto-scaling logic
        x_plot_min, x_plot_max = x.min(), x.max()
        y_reg_min = model.predict([[x_plot_min]])[0]
        y_reg_max = model.predict([[x_plot_max]])[0]
        y_reg_center = (y_reg_min + y_reg_max) / 2
        
        # Set y-axis limits centered on regression line
        y_range = percentile_high - percentile_low
        y_padding = y_range * 0.6
        
        y_limit_min = min(percentile_low, y_reg_center - y_padding)
        y_limit_max = max(percentile_high, y_reg_center + y_padding)
        
        # Ensure zero is visible if regression line crosses it
        if y_reg_min * y_reg_max < 0 or abs(y_reg_center) < y_padding:
            y_limit_min = min(y_limit_min, -y_padding * 0.5)
            y_limit_max = max(y_limit_max, y_padding * 0.5)
    
    ax.set_ylim(y_limit_min, y_limit_max)
    
    # Add subtle gradient background for positive/negative regions
    # Both metrics use zero as reference for CHANGES
    ax.axhspan(0, y_limit_max*1.5, alpha=0.02, color=COLORS['success_green'], zorder=0)
    ax.axhspan(y_limit_min*1.5, 0, alpha=0.02, color=COLORS['danger_red'], zorder=0)
    # Zero line with enhanced styling
    ax.axhline(y=0, color=COLORS['dark_gray'], linestyle='-', linewidth=1.5, alpha=0.3, zorder=1)
    
    # Scatter plot with gradient colors based on value
    # Mark outliers differently
    colors = []
    alphas = []
    edgecolors = []
    
    # For CHANGE plots, zero is always the baseline
    baseline = 0
    
    for i, val in enumerate(y):
        is_outlier = val < percentile_low or val > percentile_high
        
        # Color based on positive/negative change
        if val > baseline:
            colors.append(COLORS['success_green'])
        else:
            colors.append(COLORS['danger_red'])
        
        if is_outlier:
            # Outliers: more transparent, no edge
            alphas.append(0.3)
            edgecolors.append('none')
        else:
            # Normal points
            alphas.append(0.7)
            edgecolors.append('white')
    
    # Use uniform size for all points
    point_size = 80
    
    for i in range(len(x)):
        ax.scatter(x[i], y[i], c=[colors[i]], alpha=alphas[i], s=point_size, 
                  edgecolors=edgecolors[i], linewidth=1.5 if edgecolors[i]=='white' else 0, 
                  zorder=3)
    
    # Regression line with confidence interval
    x_smooth = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
    y_smooth = model.predict(x_smooth)
    
    # Calculate standard error for confidence interval
    residuals = y - y_pred
    std_error = np.sqrt(np.sum(residuals**2) / (n - 2))
    margin = 1.96 * std_error  # 95% CI
    
    
    # Plot regression line
    ax.plot(x_smooth, y_smooth, color=params['color'], linewidth=3, 
           zorder=4)
    
    # Enhanced formatting
    ax.set_xlabel('Follow-up time (months)', fontsize=14, fontweight='medium')
    ax.set_ylabel(params['ylabel'], fontsize=14, fontweight='medium')
    ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Add statistics text box
    n_positive = (y > 0).sum()
    pct_positive = n_positive / len(y) * 100
    
    # Determine significance
    if p_value < 0.001:
        sig_text = "***"
    elif p_value < 0.01:
        sig_text = "**"
    elif p_value < 0.05:
        sig_text = "*"
    else:
        sig_text = ""
    
    # Format slope with appropriate precision
    slope_text = f"{model.coef_[0]:.4f}" if metric == 'mld' else f"{model.coef_[0]:.3f}"
    
    # Create stats text
    stats_text = (
        f"Slope = {slope_text}/month\n"
        f"p = {p_value:.3f} {sig_text}"
    )
    
    # Add statistics box similar to paired analysis
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.95,
                     edgecolor=params['color'], linewidth=1.5))
    
    # Add subtle trend indicator if significant (very light)
    if p_value < 0.05:
        if model.coef_[0] > 0:
            trend_symbol = "↑"
            ax.text(0.95, 0.95, trend_symbol, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   fontsize=36, fontweight='bold', color=params['color'], alpha=0.08)
        else:
            trend_symbol = "↓"
            ax.text(0.95, 0.05, trend_symbol, transform=ax.transAxes,
                   verticalalignment='bottom', horizontalalignment='right',
                   fontsize=36, fontweight='bold', color=params['color'], alpha=0.08)
    
    plt.tight_layout()
    
    # Save
    filename = f'stats/plots/temporal/{metric}_time_regression.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"Saved: {filename}")
    
    return {'n': len(data), 'slope': model.coef_[0], 'p_value': p_value, 'r_squared': r_squared}

def save_summary_statistics(results):
    """Save summary statistics to CSV with enhanced formatting"""
    
    # Flatten the results for CSV
    summary_data = []
    
    if 'paired' in results:
        for metric in ['mld', 'mufr']:
            if metric in results['paired']:
                summary_data.append({
                    'analysis': 'paired',
                    'metric': metric.upper(),
                    'n': results['paired'][metric]['n'],
                    'value': results['paired'][metric]['mean_change'],
                    'p_value': results['paired'][metric]['p_value']
                })
    
    for metric in ['mld', 'mufr']:
        key = f'{metric}_regression'
        if key in results:
            summary_data.append({
                'analysis': 'time_regression',
                'metric': metric.upper(),
                'n': results[key]['n'],
                'value': results[key]['slope'],
                'p_value': results[key]['p_value'],
                'r_squared': results[key].get('r_squared', None)
            })
    
    summary_df = pd.DataFrame(summary_data)
    filename = 'stats/results/temporal/analysis_summary.csv'
    summary_df.to_csv(filename, index=False)
    print(f"\nSaved summary to: {filename}")
    
    # Print enhanced summary
    print("\n" + "═"*60)
    print(" "*20 + "ANALYSIS SUMMARY")
    print("═"*60)
    
    print("\n📊 PAIRED ANALYSIS (Post-procedural → Follow-up):")
    print("─"*60)
    if 'paired' in results:
        for metric in ['mld', 'mufr']:
            if metric in results['paired']:
                r = results['paired'][metric]
                sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
                print(f"  {metric.upper():4s}: n={r['n']:3d} | Δ={r['mean_change']:+.4f} | p={r['p_value']:.4f} {sig}")
    
    print("\n📈 TIME REGRESSION ANALYSIS:")
    print("─"*60)
    for metric in ['mld', 'mufr']:
        key = f'{metric}_regression'
        if key in results:
            r = results[key]
            sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else "ns"
            print(f"  {metric.upper():4s}: n={r['n']:3d} | slope={r['slope']:+.5f} | R²={r.get('r_squared', 0):.3f} | p={r['p_value']:.4f} {sig}")

def main():
    """Main execution function"""
    
    print("\n" + "═"*60)
    print(" "*15 + "DCB ANALYSIS: ENHANCED PLOTS")
    print("═"*60)
    
    # Setup
    create_directories()
    
    # Connect to database
    engine = connect_to_db()
    if engine is None:
        print("Failed to connect to database")
        return
    
    # Load data
    print("\n🔄 Loading data...")
    df = load_data(engine)
    print(f"✅ Loaded {len(df)} records")
    
    results = {}
    
    # PLOT 1: Combined paired analysis (MLD and μFR)
    print("\n📊 Creating combined paired analysis plot...")
    paired_results = plot_combined_paired_analysis(df)
    if paired_results:
        results['paired'] = paired_results
    
    # PLOT 2: MLD late change over time
    print("\n📈 Creating MLD late change regression plot...")
    mld_regression = plot_time_regression(df, 'mld')
    if mld_regression:
        results['mld_regression'] = mld_regression
    
    # PLOT 3: μFR late change over time
    print("\n📈 Creating μFR late change regression plot...")
    mufr_regression = plot_time_regression(df, 'mufr')
    if mufr_regression:
        results['mufr_regression'] = mufr_regression
    
    # Save summary
    if results:
        save_summary_statistics(results)
    
    print("\n" + "═"*60)
    print(" "*20 + "ANALYSIS COMPLETE")
    print("═"*60)
    print("\n✅ Plots generated:")
    print("  📊 stats/plots/temporal/combined_paired_analysis.png")
    print("  📈 stats/plots/temporal/mld_time_regression.png")
    print("  📈 stats/plots/temporal/mufr_time_regression.png")
    print("\n📄 Summary: stats/results/temporal/analysis_summary.csv")
    
    return df

if __name__ == "__main__":
    df = main()