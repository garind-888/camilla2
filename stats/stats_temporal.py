import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel
from sqlalchemy import create_engine
import os
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'dcb',
    'user': 'doriangarin',
    'password': '96349130dG!',
    'port': 5432
}

def create_directories():
    """Create directories for saving results and plots"""
    os.makedirs('stats/plots/temporal', exist_ok=True)
    os.makedirs('stats/results/temporal', exist_ok=True)
    print("Created directories: stats/plots/temporal and stats/results/temporal")

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
    """Load data with temporal information"""
    query = """
    SELECT 
        -- Temporal variables
        date_dcb as date_exam,  -- Initial procedure date
        date_fup_coro,          -- Follow-up date
        
        -- MLD measurements
        init_pre_mld,
        init_post_mld,
        fup_pre_mld,
        fup_post_mld,
        
        -- MLA measurements
        init_pre_mla,
        init_post_mla,
        fup_pre_mla,
        fup_post_mla,
        
        -- μFR measurements
        init_pre_mufr,
        init_post_mufr,
        fup_pre_mufr,
        fup_post_mufr,
        
        -- Calculated changes
        mld_late_lumen_change,
        mla_late_lumen_change,
        mufr_late_functional_change,
        
        -- Clinical outcomes
        tvf_fup_coro,
        tlf_fup_coro,
        is_residual_stenosis_at_fup,
        
        -- Patient/lesion identifier (add if available)
        ROW_NUMBER() OVER (ORDER BY date_dcb) as lesion_id
        
    FROM camilla2
    WHERE date_dcb IS NOT NULL 
      AND date_fup_coro IS NOT NULL
      AND tlf_fup_coro = 0
      AND date_fup_coro > date_dcb + INTERVAL '7 days'
      
    """
    
    df = pd.read_sql(query, engine)
    
    # Convert dates to datetime
    df['date_exam'] = pd.to_datetime(df['date_exam'])
    df['date_fup_coro'] = pd.to_datetime(df['date_fup_coro'])
    
    # Calculate follow-up duration
    df['days_to_followup'] = (df['date_fup_coro'] - df['date_exam']).dt.days
    df['months_to_followup'] = df['days_to_followup'] / 30.44
    
    return df

def perform_paired_analysis(df, var_post, var_fup, var_name, timepoint_comparison):
    """
    Perform paired analysis comparing post-procedure to follow-up
    
    Parameters:
    -----------
    df : DataFrame
    var_post : str - column name for post-procedure measurement
    var_fup : str - column name for follow-up measurement
    var_name : str - descriptive name of the variable
    timepoint_comparison : str - description of comparison (e.g., "Init Post → FUP Pre")
    """
    
    # Get paired data (only complete pairs)
    paired_data = df[[var_post, var_fup, 'months_to_followup']].dropna()
    n_pairs = len(paired_data)
    
    if n_pairs < 3:
        print(f"Insufficient paired data for {var_name}")
        return None
    
    # Calculate differences
    differences = paired_data[var_fup] - paired_data[var_post]
    
    # Perform normality test
    _, p_normal = stats.shapiro(differences) if len(differences) < 5000 else stats.normaltest(differences)
    
    # Choose appropriate test based on normality
    if p_normal > 0.05:
        # Use paired t-test for normal data
        t_stat, p_value = stats.ttest_rel(paired_data[var_fup], paired_data[var_post])
        test_used = "Paired t-test"
        test_statistic = f"t = {t_stat:.3f}"
    else:
        # Use Wilcoxon signed-rank test for non-normal data
        stat, p_value = stats.wilcoxon(paired_data[var_fup], paired_data[var_post])
        test_used = "Wilcoxon signed-rank"
        test_statistic = f"W = {stat:.1f}"
    
    # Calculate effect size (Cohen's d)
    cohens_d = differences.mean() / differences.std() if differences.std() > 0 else 0
    
    # Calculate confidence interval for mean difference
    ci_95 = stats.t.interval(0.95, len(differences)-1, 
                             differences.mean(), 
                             differences.sem())
    
    # Determine clinical significance
    if var_name == 'MLD':
        clinically_significant = abs(differences.mean()) > 0.2  # 0.2 mm threshold
    elif var_name == 'MLA':
        clinically_significant = abs(differences.mean()) > 0.5  # 0.5 mm² threshold
    elif var_name == 'μFR':
        clinically_significant = abs(differences.mean()) > 0.05  # 0.05 threshold
    else:
        clinically_significant = False
    
    # Calculate proportion with gain/loss
    n_gain = (differences > 0).sum()
    n_loss = (differences < 0).sum()
    n_stable = (differences == 0).sum()
    
    results = {
        'Variable': var_name,
        'Comparison': timepoint_comparison,
        'N_pairs': n_pairs,
        'Baseline_mean': paired_data[var_post].mean(),
        'Baseline_SD': paired_data[var_post].std(),
        'Followup_mean': paired_data[var_fup].mean(),
        'Followup_SD': paired_data[var_fup].std(),
        'Mean_change': differences.mean(),
        'SD_change': differences.std(),
        'Median_change': differences.median(),
        'IQR_25': differences.quantile(0.25),
        'IQR_75': differences.quantile(0.75),
        'CI_95_lower': ci_95[0],
        'CI_95_upper': ci_95[1],
        'Test_used': test_used,
        'Test_statistic': test_statistic,
        'P_value': p_value,
        'Cohens_d': cohens_d,
        'N_gain': n_gain,
        'N_loss': n_loss,
        'N_stable': n_stable,
        'Percent_gain': (n_gain/n_pairs)*100,
        'Percent_loss': (n_loss/n_pairs)*100,
        'Mean_followup_months': paired_data['months_to_followup'].mean(),
        'Clinically_significant': clinically_significant
    }
    
    return results

def create_mld_temporal_plot(df):
    """Create temporal plot specifically for MLD evolution"""
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create GridSpec
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.25)
    
    # Main temporal plot
    ax_main = fig.add_subplot(gs[0, :])
    
    # Box plots
    ax_box = fig.add_subplot(gs[1, 0])
    ax_violin = fig.add_subplot(gs[1, 1])
    
    # Change distribution
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_waterfall = fig.add_subplot(gs[2, 1])
    
    # Color scheme
    colors = {
        'gain': '#27AE60',
        'loss': '#E74C3C',
        'stable': '#95A5A6',
        'mean': '#2E86C1'
    }
    
    # ========== MAIN TEMPORAL PLOT ==========
    for idx, row in df.iterrows():
        if pd.notna(row['init_post_mld']) and pd.notna(row['fup_pre_mld']):
            # Time points
            t0 = 0  # Initial pre
            t1 = 0.1  # Initial post
            t2 = row['months_to_followup']  # FUP pre
            t3 = row['months_to_followup'] + 0.1  # FUP post
            
            # MLD values
            mld_times = [t0, t1, t2]
            mld_values = [row['init_pre_mld'], row['init_post_mld'], row['fup_pre_mld']]
            
            # Add FUP post if available
            if pd.notna(row['fup_post_mld']):
                mld_times.append(t3)
                mld_values.append(row['fup_post_mld'])
            
            # Remove NaN values
            valid_mld = [(t, v) for t, v in zip(mld_times, mld_values) if pd.notna(v)]
            
            if len(valid_mld) >= 3:
                mld_times_clean, mld_values_clean = zip(*valid_mld)
                
                # Determine color based on late change
                change = mld_values_clean[2] - mld_values_clean[1]  # FUP pre - Init post
                if change > 0.1:
                    line_color = colors['gain']
                    alpha = 0.4
                    zorder = 2
                elif change < -0.1:
                    line_color = colors['loss']
                    alpha = 0.3
                    zorder = 1
                else:
                    line_color = colors['stable']
                    alpha = 0.2
                    zorder = 0
                
                ax_main.plot(mld_times_clean, mld_values_clean, 
                           color=line_color, alpha=alpha, linewidth=0.8, zorder=zorder)
    
    # Add mean trajectory
    mean_mld_pre = df['init_pre_mld'].mean()
    mean_mld_post = df['init_post_mld'].mean()
    mean_mld_fup_pre = df['fup_pre_mld'].mean()
    mean_mld_fup_post = df['fup_post_mld'].mean()
    mean_months = df['months_to_followup'].mean()
    
    mean_times = [0, 0.1, mean_months, mean_months + 0.1]
    mean_mld = [mean_mld_pre, mean_mld_post, mean_mld_fup_pre, mean_mld_fup_post]
    
    # Plot mean trajectory with markers
    ax_main.plot(mean_times[:3], mean_mld[:3], 
                color=colors['mean'], linewidth=3, marker='o', 
                markersize=12, label='Mean MLD', zorder=100)
    
    if pd.notna(mean_mld_fup_post):
        ax_main.plot([mean_times[2], mean_times[3]], [mean_mld[2], mean_mld[3]], 
                    color=colors['mean'], linewidth=3, linestyle='--', 
                    marker='s', markersize=12, alpha=0.7, zorder=100)
    
    # Add annotations for mean values
    for t, v in zip(mean_times[:3], mean_mld[:3]):
        ax_main.annotate(f'{v:.2f}', (t, v), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color=colors['mean'])
    
    # Shaded regions
    ax_main.axvspan(-0.5, 0.5, alpha=0.1, color='blue', label='Initial Procedure')
    ax_main.axvspan(mean_months-0.5, mean_months+0.5, alpha=0.1, color='orange', label='Follow-up')
    
    # Formatting
    ax_main.set_xlabel('Time (months)', fontsize=12)
    ax_main.set_ylabel('MLD (mm)', fontsize=12)
    ax_main.set_title('MLD Temporal Evolution', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='best')
    ax_main.set_xlim(-1, max(df['months_to_followup'].max() + 1, mean_months + 2))
    
    # ========== BOX PLOT ==========
    mld_data = [
        df['init_pre_mld'].dropna(),
        df['init_post_mld'].dropna(),
        df['fup_pre_mld'].dropna(),
        df['fup_post_mld'].dropna()
    ]
    
    bp = ax_box.boxplot(mld_data, labels=['Pre', 'Post', 'FUP\nPre', 'FUP\nPost'],
                        patch_artist=True, showmeans=True)
    
    box_colors = ['#E8F8F5', '#A9DFBF', '#F9E79F', '#FAD7A0']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Add significance test
    _, p_value = stats.ttest_rel(
        df[['init_post_mld', 'fup_pre_mld']].dropna()['init_post_mld'],
        df[['init_post_mld', 'fup_pre_mld']].dropna()['fup_pre_mld']
    )
    
    sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    y_max = ax_box.get_ylim()[1]
    ax_box.plot([1.8, 2.8], [y_max*0.95, y_max*0.95], 'k-', linewidth=1)
    ax_box.text(2.3, y_max*0.97, f'{sig_text}\np={p_value:.4f}', ha='center', fontsize=9)
    
    ax_box.set_ylabel('MLD (mm)', fontsize=10)
    ax_box.set_title('MLD Distribution by Timepoint', fontsize=11)
    ax_box.grid(True, alpha=0.3, axis='y')
    
    # ========== VIOLIN PLOT ==========
    parts = ax_violin.violinplot(mld_data, positions=[1, 2, 3, 4], 
                                 widths=0.7, showmeans=True, showmedians=True)
    
    for pc, color in zip(parts['bodies'], box_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)
    
    ax_violin.set_xticks([1, 2, 3, 4])
    ax_violin.set_xticklabels(['Pre', 'Post', 'FUP\nPre', 'FUP\nPost'])
    ax_violin.set_ylabel('MLD (mm)', fontsize=10)
    ax_violin.set_title('MLD Distribution (Violin)', fontsize=11)
    ax_violin.grid(True, alpha=0.3, axis='y')
    
    # ========== CHANGE HISTOGRAM ==========
    mld_change = df['fup_pre_mld'] - df['init_post_mld']
    mld_change_clean = mld_change.dropna()
    
    ax_hist.hist(mld_change_clean, bins=20, edgecolor='black', alpha=0.7)
    ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_hist.axvline(x=mld_change_clean.mean(), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean: {mld_change_clean.mean():.3f}')
    
    # Add shaded regions for gain/loss
    ax_hist.axvspan(0, mld_change_clean.max(), alpha=0.2, color='green')
    ax_hist.axvspan(mld_change_clean.min(), 0, alpha=0.2, color='red')
    
    ax_hist.set_xlabel('MLD Change (mm)', fontsize=10)
    ax_hist.set_ylabel('Frequency', fontsize=10)
    ax_hist.set_title('Distribution of Late MLD Change', fontsize=11)
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # Add text annotations
    gain_pct = (mld_change_clean > 0).mean() * 100
    loss_pct = (mld_change_clean < 0).mean() * 100
    ax_hist.text(0.98, 0.98, f'Gain: {gain_pct:.1f}%\nLoss: {loss_pct:.1f}%', 
                transform=ax_hist.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== WATERFALL PLOT ==========
    sorted_changes = mld_change_clean.sort_values()
    x_pos = np.arange(len(sorted_changes))
    
    colors_waterfall = ['green' if c > 0 else 'red' for c in sorted_changes]
    ax_waterfall.bar(x_pos, sorted_changes, color=colors_waterfall, alpha=0.7)
    ax_waterfall.axhline(y=0, color='black', linewidth=1)
    ax_waterfall.set_xlabel('Lesion (sorted)', fontsize=10)
    ax_waterfall.set_ylabel('MLD Change (mm)', fontsize=10)
    ax_waterfall.set_title('Waterfall Plot of MLD Changes', fontsize=11)
    ax_waterfall.grid(True, alpha=0.3, axis='y')
    
    # Overall title and layout
    fig.suptitle('MLD (Minimal Lumen Diameter) Temporal Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('stats/plots/temporal/mld_temporal_evolution.png', dpi=300, bbox_inches='tight')
    
    
    print("MLD temporal plot saved to: stats/plots/temporal/mld_temporal_evolution.png")

def create_mla_temporal_plot(df):
    """Create temporal plot specifically for MLA evolution"""
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create GridSpec
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.25)
    
    # Main temporal plot
    ax_main = fig.add_subplot(gs[0, :])
    
    # Box plots
    ax_box = fig.add_subplot(gs[1, 0])
    ax_violin = fig.add_subplot(gs[1, 1])
    
    # Change distribution
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_waterfall = fig.add_subplot(gs[2, 1])
    
    # Color scheme
    colors = {
        'gain': '#27AE60',
        'loss': '#E74C3C',
        'stable': '#95A5A6',
        'mean': '#8E44AD'
    }
    
    # ========== MAIN TEMPORAL PLOT ==========
    for idx, row in df.iterrows():
        if pd.notna(row['init_post_mla']) and pd.notna(row['fup_pre_mla']):
            # Time points
            t0 = 0
            t1 = 0.1
            t2 = row['months_to_followup']
            t3 = row['months_to_followup'] + 0.1
            
            # MLA values
            mla_times = [t0, t1, t2]
            mla_values = [row['init_pre_mla'], row['init_post_mla'], row['fup_pre_mla']]
            
            if pd.notna(row['fup_post_mla']):
                mla_times.append(t3)
                mla_values.append(row['fup_post_mla'])
            
            valid_mla = [(t, v) for t, v in zip(mla_times, mla_values) if pd.notna(v)]
            
            if len(valid_mla) >= 3:
                mla_times_clean, mla_values_clean = zip(*valid_mla)
                
                # Determine color based on late change
                change = mla_values_clean[2] - mla_values_clean[1]
                if change > 0.3:
                    line_color = colors['gain']
                    alpha = 0.4
                    zorder = 2
                elif change < -0.3:
                    line_color = colors['loss']
                    alpha = 0.3
                    zorder = 1
                else:
                    line_color = colors['stable']
                    alpha = 0.2
                    zorder = 0
                
                ax_main.plot(mla_times_clean, mla_values_clean, 
                           color=line_color, alpha=alpha, linewidth=0.8, zorder=zorder)
    
    # Add mean trajectory
    mean_mla_pre = df['init_pre_mla'].mean()
    mean_mla_post = df['init_post_mla'].mean()
    mean_mla_fup_pre = df['fup_pre_mla'].mean()
    mean_mla_fup_post = df['fup_post_mla'].mean()
    mean_months = df['months_to_followup'].mean()
    
    mean_times = [0, 0.1, mean_months, mean_months + 0.1]
    mean_mla = [mean_mla_pre, mean_mla_post, mean_mla_fup_pre, mean_mla_fup_post]
    
    # Plot mean trajectory
    ax_main.plot(mean_times[:3], mean_mla[:3], 
                color=colors['mean'], linewidth=3, marker='o', 
                markersize=12, label='Mean MLA', zorder=100)
    
    if pd.notna(mean_mla_fup_post):
        ax_main.plot([mean_times[2], mean_times[3]], [mean_mla[2], mean_mla[3]], 
                    color=colors['mean'], linewidth=3, linestyle='--', 
                    marker='s', markersize=12, alpha=0.7, zorder=100)
    
    # Add annotations
    for t, v in zip(mean_times[:3], mean_mla[:3]):
        ax_main.annotate(f'{v:.2f}', (t, v), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color=colors['mean'])
    
    # Shaded regions
    ax_main.axvspan(-0.5, 0.5, alpha=0.1, color='blue', label='Initial Procedure')
    ax_main.axvspan(mean_months-0.5, mean_months+0.5, alpha=0.1, color='orange', label='Follow-up')
    
    # Formatting
    ax_main.set_xlabel('Time (months)', fontsize=12)
    ax_main.set_ylabel('MLA (mm²)', fontsize=12)
    ax_main.set_title('MLA Temporal Evolution', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='best')
    ax_main.set_xlim(-1, max(df['months_to_followup'].max() + 1, mean_months + 2))
    
    # ========== BOX PLOT ==========
    mla_data = [
        df['init_pre_mla'].dropna(),
        df['init_post_mla'].dropna(),
        df['fup_pre_mla'].dropna(),
        df['fup_post_mla'].dropna()
    ]
    
    bp = ax_box.boxplot(mla_data, labels=['Pre', 'Post', 'FUP\nPre', 'FUP\nPost'],
                        patch_artist=True, showmeans=True)
    
    box_colors = ['#EBDEF0', '#D7BDE2', '#FADBD8', '#F6DDCC']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Add significance test
    _, p_value = stats.ttest_rel(
        df[['init_post_mla', 'fup_pre_mla']].dropna()['init_post_mla'],
        df[['init_post_mla', 'fup_pre_mla']].dropna()['fup_pre_mla']
    )
    
    sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    y_max = ax_box.get_ylim()[1]
    ax_box.plot([1.8, 2.8], [y_max*0.95, y_max*0.95], 'k-', linewidth=1)
    ax_box.text(2.3, y_max*0.97, f'{sig_text}\np={p_value:.4f}', ha='center', fontsize=9)
    
    ax_box.set_ylabel('MLA (mm²)', fontsize=10)
    ax_box.set_title('MLA Distribution by Timepoint', fontsize=11)
    ax_box.grid(True, alpha=0.3, axis='y')
    
    # ========== VIOLIN PLOT ==========
    parts = ax_violin.violinplot(mla_data, positions=[1, 2, 3, 4], 
                                 widths=0.7, showmeans=True, showmedians=True)
    
    for pc, color in zip(parts['bodies'], box_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)
    
    ax_violin.set_xticks([1, 2, 3, 4])
    ax_violin.set_xticklabels(['Pre', 'Post', 'FUP\nPre', 'FUP\nPost'])
    ax_violin.set_ylabel('MLA (mm²)', fontsize=10)
    ax_violin.set_title('MLA Distribution (Violin)', fontsize=11)
    ax_violin.grid(True, alpha=0.3, axis='y')
    
    # ========== CHANGE HISTOGRAM ==========
    mla_change = df['fup_pre_mla'] - df['init_post_mla']
    mla_change_clean = mla_change.dropna()
    
    ax_hist.hist(mla_change_clean, bins=20, edgecolor='black', alpha=0.7)
    ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_hist.axvline(x=mla_change_clean.mean(), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean: {mla_change_clean.mean():.3f}')
    
    # Add shaded regions
    ax_hist.axvspan(0, mla_change_clean.max(), alpha=0.2, color='green')
    ax_hist.axvspan(mla_change_clean.min(), 0, alpha=0.2, color='red')
    
    ax_hist.set_xlabel('MLA Change (mm²)', fontsize=10)
    ax_hist.set_ylabel('Frequency', fontsize=10)
    ax_hist.set_title('Distribution of Late MLA Change', fontsize=11)
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # Add text annotations
    gain_pct = (mla_change_clean > 0).mean() * 100
    loss_pct = (mla_change_clean < 0).mean() * 100
    ax_hist.text(0.98, 0.98, f'Gain: {gain_pct:.1f}%\nLoss: {loss_pct:.1f}%', 
                transform=ax_hist.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== WATERFALL PLOT ==========
    sorted_changes = mla_change_clean.sort_values()
    x_pos = np.arange(len(sorted_changes))
    
    colors_waterfall = ['green' if c > 0 else 'red' for c in sorted_changes]
    ax_waterfall.bar(x_pos, sorted_changes, color=colors_waterfall, alpha=0.7)
    ax_waterfall.axhline(y=0, color='black', linewidth=1)
    ax_waterfall.set_xlabel('Lesion (sorted)', fontsize=10)
    ax_waterfall.set_ylabel('MLA Change (mm²)', fontsize=10)
    ax_waterfall.set_title('Waterfall Plot of MLA Changes', fontsize=11)
    ax_waterfall.grid(True, alpha=0.3, axis='y')
    
    # Overall title and layout
    fig.suptitle('MLA (Minimal Lumen Area) Temporal Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('stats/plots/temporal/mla_temporal_evolution.png', dpi=300, bbox_inches='tight')
    
    
    print("MLA temporal plot saved to: stats/plots/temporal/mla_temporal_evolution.png")

def create_mufr_temporal_plot(df):
    """Create temporal plot specifically for μFR evolution"""
    
    fig = plt.figure(figsize=(14, 10))
    
    # Create GridSpec
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.25)
    
    # Main temporal plot
    ax_main = fig.add_subplot(gs[0, :])
    
    # Box plots
    ax_box = fig.add_subplot(gs[1, 0])
    ax_violin = fig.add_subplot(gs[1, 1])
    
    # Change distribution
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_waterfall = fig.add_subplot(gs[2, 1])
    
    # Color scheme
    colors = {
        'gain': '#27AE60',
        'loss': '#E74C3C',
        'stable': '#95A5A6',
        'mean': '#E67E22'
    }
    
    # ========== MAIN TEMPORAL PLOT ==========
    for idx, row in df.iterrows():
        if pd.notna(row['init_post_mufr']) and pd.notna(row['fup_pre_mufr']):
            # Time points
            t0 = 0
            t1 = 0.1
            t2 = row['months_to_followup']
            t3 = row['months_to_followup'] + 0.1
            
            # μFR values
            mufr_times = [t0, t1, t2]
            mufr_values = [row['init_pre_mufr'], row['init_post_mufr'], row['fup_pre_mufr']]
            
            if pd.notna(row['fup_post_mufr']):
                mufr_times.append(t3)
                mufr_values.append(row['fup_post_mufr'])
            
            valid_mufr = [(t, v) for t, v in zip(mufr_times, mufr_values) if pd.notna(v)]
            
            if len(valid_mufr) >= 3:
                mufr_times_clean, mufr_values_clean = zip(*valid_mufr)
                
                # Determine color based on late change
                change = mufr_values_clean[2] - mufr_values_clean[1]
                if change > 0.03:
                    line_color = colors['gain']
                    alpha = 0.4
                    zorder = 2
                elif change < -0.03:
                    line_color = colors['loss']
                    alpha = 0.3
                    zorder = 1
                else:
                    line_color = colors['stable']
                    alpha = 0.2
                    zorder = 0
                
                ax_main.plot(mufr_times_clean, mufr_values_clean, 
                           color=line_color, alpha=alpha, linewidth=0.8, zorder=zorder)
    
    # Add mean trajectory
    mean_mufr_pre = df['init_pre_mufr'].mean()
    mean_mufr_post = df['init_post_mufr'].mean()
    mean_mufr_fup_pre = df['fup_pre_mufr'].mean()
    mean_mufr_fup_post = df['fup_post_mufr'].mean()
    mean_months = df['months_to_followup'].mean()
    
    mean_times = [0, 0.1, mean_months, mean_months + 0.1]
    mean_mufr = [mean_mufr_pre, mean_mufr_post, mean_mufr_fup_pre, mean_mufr_fup_post]
    
    # Plot mean trajectory
    ax_main.plot(mean_times[:3], mean_mufr[:3], 
                color=colors['mean'], linewidth=3, marker='o', 
                markersize=12, label='Mean μFR', zorder=100)
    
    if pd.notna(mean_mufr_fup_post):
        ax_main.plot([mean_times[2], mean_times[3]], [mean_mufr[2], mean_mufr[3]], 
                    color=colors['mean'], linewidth=3, linestyle='--', 
                    marker='s', markersize=12, alpha=0.7, zorder=100)
    
    # Add annotations
    for t, v in zip(mean_times[:3], mean_mufr[:3]):
        ax_main.annotate(f'{v:.3f}', (t, v), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, color=colors['mean'])
    
    # Add reference lines for μFR cutoffs
    ax_main.axhline(y=0.80, color='red', linestyle=':', alpha=0.5, label='μFR = 0.80')
    ax_main.axhline(y=0.90, color='orange', linestyle=':', alpha=0.5, label='μFR = 0.90')
    
    # Shaded regions
    ax_main.axvspan(-0.5, 0.5, alpha=0.1, color='blue', label='Initial Procedure')
    ax_main.axvspan(mean_months-0.5, mean_months+0.5, alpha=0.1, color='orange', label='Follow-up')
    
    # Formatting
    ax_main.set_xlabel('Time (months)', fontsize=12)
    ax_main.set_ylabel('μFR', fontsize=12)
    ax_main.set_title('μFR Temporal Evolution', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend(loc='best')
    ax_main.set_xlim(-1, max(df['months_to_followup'].max() + 1, mean_months + 2))
    ax_main.set_ylim(0.5, 1.05)
    
    # ========== BOX PLOT ==========
    mufr_data = [
        df['init_pre_mufr'].dropna(),
        df['init_post_mufr'].dropna(),
        df['fup_pre_mufr'].dropna(),
        df['fup_post_mufr'].dropna()
    ]
    
    bp = ax_box.boxplot(mufr_data, labels=['Pre', 'Post', 'FUP\nPre', 'FUP\nPost'],
                        patch_artist=True, showmeans=True)
    
    box_colors = ['#FEF5E7', '#FCF3CF', '#F9E79F', '#F8C471']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    # Add significance test
    _, p_value = stats.ttest_rel(
        df[['init_post_mufr', 'fup_pre_mufr']].dropna()['init_post_mufr'],
        df[['init_post_mufr', 'fup_pre_mufr']].dropna()['fup_pre_mufr']
    )
    
    sig_text = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    y_max = ax_box.get_ylim()[1]
    ax_box.plot([1.8, 2.8], [y_max*0.95, y_max*0.95], 'k-', linewidth=1)
    ax_box.text(2.3, y_max*0.97, f'{sig_text}\np={p_value:.4f}', ha='center', fontsize=9)
    
    # Add reference lines
    ax_box.axhline(y=0.80, color='red', linestyle=':', alpha=0.5)
    ax_box.axhline(y=0.90, color='orange', linestyle=':', alpha=0.5)
    
    ax_box.set_ylabel('μFR', fontsize=10)
    ax_box.set_title('μFR Distribution by Timepoint', fontsize=11)
    ax_box.grid(True, alpha=0.3, axis='y')
    
    # ========== VIOLIN PLOT ==========
    parts = ax_violin.violinplot(mufr_data, positions=[1, 2, 3, 4], 
                                 widths=0.7, showmeans=True, showmedians=True)
    
    for pc, color in zip(parts['bodies'], box_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.8)
    
    # Add reference lines
    ax_violin.axhline(y=0.80, color='red', linestyle=':', alpha=0.5)
    ax_violin.axhline(y=0.90, color='orange', linestyle=':', alpha=0.5)
    
    ax_violin.set_xticks([1, 2, 3, 4])
    ax_violin.set_xticklabels(['Pre', 'Post', 'FUP\nPre', 'FUP\nPost'])
    ax_violin.set_ylabel('μFR', fontsize=10)
    ax_violin.set_title('μFR Distribution (Violin)', fontsize=11)
    ax_violin.grid(True, alpha=0.3, axis='y')
    
    # ========== CHANGE HISTOGRAM ==========
    mufr_change = df['fup_pre_mufr'] - df['init_post_mufr']
    mufr_change_clean = mufr_change.dropna()
    
    ax_hist.hist(mufr_change_clean, bins=20, edgecolor='black', alpha=0.7)
    ax_hist.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax_hist.axvline(x=mufr_change_clean.mean(), color='blue', linestyle='-', linewidth=2, 
                   label=f'Mean: {mufr_change_clean.mean():.3f}')
    ax_hist.axvline(x=0.05, color='green', linestyle=':', linewidth=1, alpha=0.7,
                   label='Clinical threshold (0.05)')
    
    # Add shaded regions
    ax_hist.axvspan(0, mufr_change_clean.max(), alpha=0.2, color='green')
    ax_hist.axvspan(mufr_change_clean.min(), 0, alpha=0.2, color='red')
    
    ax_hist.set_xlabel('μFR Change', fontsize=10)
    ax_hist.set_ylabel('Frequency', fontsize=10)
    ax_hist.set_title('Distribution of Late μFR Change', fontsize=11)
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # Add text annotations
    gain_pct = (mufr_change_clean > 0).mean() * 100
    loss_pct = (mufr_change_clean < 0).mean() * 100
    ax_hist.text(0.98, 0.98, f'Gain: {gain_pct:.1f}%\nLoss: {loss_pct:.1f}%', 
                transform=ax_hist.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== WATERFALL PLOT ==========
    sorted_changes = mufr_change_clean.sort_values()
    x_pos = np.arange(len(sorted_changes))
    
    colors_waterfall = ['green' if c > 0 else 'red' for c in sorted_changes]
    ax_waterfall.bar(x_pos, sorted_changes, color=colors_waterfall, alpha=0.7)
    ax_waterfall.axhline(y=0, color='black', linewidth=1)
    ax_waterfall.axhline(y=0.05, color='green', linestyle=':', alpha=0.5)
    ax_waterfall.axhline(y=-0.05, color='red', linestyle=':', alpha=0.5)
    ax_waterfall.set_xlabel('Lesion (sorted)', fontsize=10)
    ax_waterfall.set_ylabel('μFR Change', fontsize=10)
    ax_waterfall.set_title('Waterfall Plot of μFR Changes', fontsize=11)
    ax_waterfall.grid(True, alpha=0.3, axis='y')
    
    # Overall title and layout
    fig.suptitle('μFR (Microvascular Fractional Flow Reserve) Temporal Analysis', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('stats/plots/temporal/mufr_temporal_evolution.png', dpi=300, bbox_inches='tight')
    
    
    print("μFR temporal plot saved to: stats/plots/temporal/mufr_temporal_evolution.png")

def create_summary_plot(df, results_df):
    """Create a summary plot showing key findings"""
    
    fig = plt.figure(figsize=(14, 8))
    
    # Create GridSpec
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Panel 1: Mean changes for each metric
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Get primary results (Init Post → FUP Pre)
    primary_results = results_df[results_df['Comparison'] == 'Init Post → FUP Pre']
    
    if not primary_results.empty:
        metrics = primary_results['Variable'].values
        mean_changes = primary_results['Mean_change'].values
        ci_lower = primary_results['CI_95_lower'].values
        ci_upper = primary_results['CI_95_upper'].values
        
        y_pos = np.arange(len(metrics))
        colors_bar = ['green' if c > 0 else 'red' for c in mean_changes]
        
        ax1.barh(y_pos, mean_changes, xerr=[mean_changes - ci_lower, ci_upper - mean_changes],
                color=colors_bar, alpha=0.7, capsize=5)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(metrics)
        ax1.set_xlabel('Mean Change (95% CI)')
        ax1.set_title('Late Changes (Init Post → FUP Pre)')
        ax1.grid(True, alpha=0.3, axis='x')
    
    # Panel 2: Proportion with gain/loss
    ax2 = fig.add_subplot(gs[0, 1])
    
    if not primary_results.empty:
        metrics = primary_results['Variable'].values
        gain_pct = primary_results['Percent_gain'].values
        loss_pct = primary_results['Percent_loss'].values
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, gain_pct, width, label='Gain', color='green', alpha=0.7)
        bars2 = ax2.bar(x + width/2, loss_pct, width, label='Loss', color='red', alpha=0.7)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.set_ylabel('Percentage (%)')
        ax2.set_title('Distribution of Outcomes')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # Panel 3: P-values
    ax3 = fig.add_subplot(gs[0, 2])
    
    if not primary_results.empty:
        metrics = primary_results['Variable'].values
        p_values = primary_results['P_value'].values
        
        y_pos = np.arange(len(metrics))
        colors_p = ['green' if p < 0.05 else 'gray' for p in p_values]
        
        bars = ax3.barh(y_pos, -np.log10(p_values), color=colors_p, alpha=0.7)
        ax3.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        ax3.axvline(x=-np.log10(0.01), color='orange', linestyle='--', label='p=0.01')
        ax3.axvline(x=-np.log10(0.001), color='darkred', linestyle='--', label='p=0.001')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(metrics)
        ax3.set_xlabel('-log10(p-value)')
        ax3.set_title('Statistical Significance')
        ax3.legend(loc='lower right', fontsize=8)
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add p-value labels
        for i, (bar, p) in enumerate(zip(bars, p_values)):
            ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f'p={p:.4f}', va='center', fontsize=8)
    
    # Panel 4-6: Individual patient changes over time
    for i, (var_name, var_post, var_pre) in enumerate([
        ('MLD', 'init_post_mld', 'fup_pre_mld'),
        ('MLA', 'init_post_mla', 'fup_pre_mla'),
        ('μFR', 'init_post_mufr', 'fup_pre_mufr')
    ]):
        ax = fig.add_subplot(gs[1, i])
        
        if var_post in df.columns and var_pre in df.columns:
            paired_data = df[[var_post, var_pre]].dropna()
            
            if len(paired_data) > 0:
                for idx, row in paired_data.iterrows():
                    change = row[var_pre] - row[var_post]
                    color = 'green' if change > 0 else 'red'
                    ax.plot([0, 1], [row[var_post], row[var_pre]], 
                           color=color, alpha=0.3, linewidth=0.8)
                
                # Add mean line
                ax.plot([0, 1], [paired_data[var_post].mean(), paired_data[var_pre].mean()],
                       'b-', linewidth=3, label='Mean', zorder=100)
                
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Init Post', 'FUP Pre'])
                ax.set_ylabel(var_name)
                ax.set_title(f'{var_name} Individual Changes')
                ax.legend()
                ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('Summary of Temporal Analysis Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('stats/plots/temporal/summary_temporal_analysis.png', dpi=300, bbox_inches='tight')
    
    
    print("Summary plot saved to: stats/plots/temporal/summary_temporal_analysis.png")

def generate_paired_analysis_report(df):
    """Generate comprehensive report of paired analyses"""
    
    print("="*80)
    print("PAIRED ANALYSIS: LATE LUMEN AND PHYSIOLOGICAL CHANGES")
    print("="*80)
    print(f"\nTotal lesions analyzed: {len(df)}")
    print(f"Mean follow-up: {df['months_to_followup'].mean():.1f} ± {df['months_to_followup'].std():.1f} months")
    print(f"Median follow-up: {df['months_to_followup'].median():.1f} months (IQR: {df['months_to_followup'].quantile(0.25):.1f}-{df['months_to_followup'].quantile(0.75):.1f})")
    
    # List of comparisons to perform
    comparisons = [
        # Init Post → FUP Pre comparisons
        ('init_post_mld', 'fup_pre_mld', 'MLD', 'TLF show'),
        ('init_post_mla', 'fup_pre_mla', 'MLA', 'TLF show'),
        ('init_post_mufr', 'fup_pre_mufr', 'μFR', 'TLF show'),
        
        # Init Post → FUP Post comparisons (if re-intervention done)
        ('init_post_mld', 'fup_post_mld', 'MLD', 'TLF ignore'),
        ('init_post_mla', 'fup_post_mla', 'MLA', 'TLF ignore'),
        ('init_post_mufr', 'fup_post_mufr', 'μFR', 'TLF ignore'),
        
    
    ]
    
    all_results = []
    
    for var_post, var_fup, var_name, comparison in comparisons:
        if var_post in df.columns and var_fup in df.columns:
            results = perform_paired_analysis(df, var_post, var_fup, var_name, comparison)
            if results:
                all_results.append(results)
    
    # Convert to DataFrame for better display
    results_df = pd.DataFrame(all_results)
    
    # Print detailed results for main comparisons (Init Post → FUP Pre)
    print("\n" + "="*80)
    print("PRIMARY ANALYSIS: INIT POST → FUP PRE")
    print("="*80)
    
    primary_results = results_df[results_df['Comparison'] == 'Init Post → FUP Pre']
    
    for _, row in primary_results.iterrows():
        print(f"\n{row['Variable']} Analysis:")
        print("-"*40)
        print(f"  Sample size: {row['N_pairs']} paired measurements")
        print(f"  Baseline (Post): {row['Baseline_mean']:.3f} ± {row['Baseline_SD']:.3f}")
        print(f"  Follow-up (Pre): {row['Followup_mean']:.3f} ± {row['Followup_SD']:.3f}")
        print(f"  Mean change: {row['Mean_change']:.3f} ± {row['SD_change']:.3f}")
        print(f"  95% CI: ({row['CI_95_lower']:.3f}, {row['CI_95_upper']:.3f})")
        print(f"  Lesions with gain: {row['N_gain']} ({row['Percent_gain']:.1f}%)")
        print(f"  Lesions with loss: {row['N_loss']} ({row['Percent_loss']:.1f}%)")
        
        # Interpretation
        if row['P_value'] < 0.05:
            direction = "INCREASE" if row['Mean_change'] > 0 else "DECREASE"
            print(f"  → Significant {direction} in {row['Variable']} at follow-up (p < 0.05)")
            if row['Clinically_significant']:
                print(f"  → Change is CLINICALLY SIGNIFICANT")
        else:
            print(f"  → No significant change in {row['Variable']} (p = {row['P_value']:.3f})")
    
    # Save results to CSV
    results_df.to_csv('stats/results/temporal/paired_analysis_results.csv', index=False)
    print("\n" + "="*80)
    print("Results saved to 'stats/results/temporal/paired_analysis_results.csv'")
    
    # Clinical summary
    print("\n" + "="*80)
    print("CLINICAL SUMMARY")
    print("="*80)
    
    # Check for late lumen gain
    mld_result = primary_results[primary_results['Variable'] == 'MLD'].iloc[0] if len(primary_results[primary_results['Variable'] == 'MLD']) > 0 else None
    if mld_result is not None:
        if mld_result['Mean_change'] > 0 and mld_result['P_value'] < 0.05:
            print("✓ Evidence of LATE LUMEN GAIN")
            print(f"  - Mean MLD increased by {mld_result['Mean_change']:.3f} mm")
            print(f"  - {mld_result['Percent_gain']:.1f}% of lesions showed gain")
        elif mld_result['Mean_change'] < 0 and mld_result['P_value'] < 0.05:
            print("✗ Evidence of LATE LUMEN LOSS")
            print(f"  - Mean MLD decreased by {abs(mld_result['Mean_change']):.3f} mm")
            print(f"  - {mld_result['Percent_loss']:.1f}% of lesions showed loss")
        else:
            print("○ No significant late lumen change")
    
    # Check for physiological gain
    mufr_result = primary_results[primary_results['Variable'] == 'μFR'].iloc[0] if len(primary_results[primary_results['Variable'] == 'μFR']) > 0 else None
    if mufr_result is not None:
        if mufr_result['Mean_change'] > 0 and mufr_result['P_value'] < 0.05:
            print("\n✓ Evidence of LATE PHYSIOLOGICAL GAIN")
            print(f"  - Mean μFR increased by {mufr_result['Mean_change']:.3f}")
            print(f"  - {mufr_result['Percent_gain']:.1f}% of lesions showed improvement")
        elif mufr_result['Mean_change'] < 0 and mufr_result['P_value'] < 0.05:
            print("\n✗ Evidence of LATE PHYSIOLOGICAL DETERIORATION")
            print(f"  - Mean μFR decreased by {abs(mufr_result['Mean_change']):.3f}")
            print(f"  - {mufr_result['Percent_loss']:.1f}% of lesions showed deterioration")
        else:
            print("\n○ No significant physiological change")
    
    return results_df

def main():
    """Main execution function"""
    
    print("Starting Paired Temporal Analysis of DCB Lesions...")
    print("="*80)
    
    # Create directories
    create_directories()
    
    # Connect to database
    engine = connect_to_db()
    if engine is None:
        print("Failed to connect to database. Please check connection parameters.")
        return
    
    # Load data
    print("\nLoading data...")
    df = load_data(engine)
    print(f"Loaded {len(df)} lesions with temporal data")
    
    # Generate paired analysis report
    print("\nPerforming paired analyses...")
    results = generate_paired_analysis_report(df)
    
    # Create individual temporal plots
    print("\nGenerating individual temporal evolution plots...")
    
    # MLD plot
    create_mld_temporal_plot(df)
    
    # MLA plot
    create_mla_temporal_plot(df)
    
    # μFR plot
    create_mufr_temporal_plot(df)
    
    # Summary plot
    create_summary_plot(df, results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    print("\nKey Questions Answered:")
    print("1. Is there late lumen gain within our patients? → Check CLINICAL SUMMARY above")
    print("2. Is there late physiological gain? → Check CLINICAL SUMMARY above")
    print("3. How do MLD, MLA, and μFR evolve over time? → See individual plots in stats/plots/temporal/")
    print("4. What proportion of patients show gain vs loss? → See detailed statistics above")
    
    print("\nOutputs generated:")
    print("- CSV Results: stats/results/temporal/paired_analysis_results.csv")
    print("- Plots saved to stats/plots/temporal/:")
    print("  • mld_temporal_evolution.png")
    print("  • mla_temporal_evolution.png")
    print("  • mufr_temporal_evolution.png")
    print("  • summary_temporal_analysis.png")
    
    return df, results

if __name__ == "__main__":
    df, results = main()