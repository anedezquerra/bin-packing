import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import shapiro, probplot, zscore
from statsmodels.stats.diagnostic import het_breuschpagan
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import gaussian_kde



def container_efficiency_analysis(data, dv='packing_ratio', group_var='container',
                                  covariates=['softness', 'items'], alpha=0.05, 
                                  plot=True, palette='viridis'):
    """
    Analyzes container efficiency using ANCOVA.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset with container, packing_ratio, etc.
    dv : str, optional
        Dependent variable (default: 'packing_ratio')
    group_var : str, optional
        Grouping variable (default: 'container')
    covariates : list, optional
        Covariates for adjustment (default: ['softness', 'items'])
    alpha : float, optional
        Significance level (default: 0.05)
    plot : bool, optional
        Generate visualization (default: True)
    palette : str, optional
        Color palette for plots (default: 'viridis')
    
    Returns:
    tuple: (ANCOVA model object, pairwise comparisons)
    """
    # Ensure data has required columns
    required = [dv, group_var] + covariates
    if not all(col in data.columns for col in required):
        raise ValueError(f"Missing required columns: {required}")
    
    # Create formula for ANCOVA
    covar_terms = " + ".join(covariates)
    formula = f"{dv} ~ C({group_var}) + {covar_terms}"
    
    # Fit ANCOVA model
    model = smf.ols(formula, data=data).fit()
    
    # Pairwise comparisons
    tukey = pairwise_tukeyhsd(data[dv], data[group_var], alpha=alpha)
    
    # Visualization
    if plot:
        plt.figure(figsize=(12, 7))
        sns.boxplot(x=group_var, y=dv, data=data, palette=palette, showmeans=True,
                    meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"black"})
        plt.title(f"{dv} Distribution by {group_var}", fontsize=14)
        plt.ylabel(dv, fontsize=12)
        plt.xlabel(group_var, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
    return model, tukey

def softness_threshold_detection(data, dv='packing_ratio', x_var='softness',
                                 group_vars=None, min_size=5, pen=2, 
                                 model_type='l2', plot=True):
    """
    Detects softness thresholds using breakpoint analysis.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    dv : str, optional
        Dependent variable (default: 'packing_ratio')
    x_var : str, optional
        Independent variable (default: 'softness')
    group_vars : list, optional
        Grouping variables for segmentation (default: None)
    min_size : int, optional
        Minimum segment size (default: 5)
    pen : float, optional
        Penalty parameter for breakpoints (default: 2)
    model_type : str, optional
        Breakpoint model type ('l1', 'l2', 'rbf') (default: 'l2')
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    dict: Breakpoint results by group
    """
    from ruptures import Binseg
    
    results = {}
    df = data.copy()
    
    # Handle grouping
    if group_vars:
        groups = df.groupby(group_vars)
    else:
        groups = [('all_data', df)]
    
    for group_key, group_df in groups:
        # Prepare data
        X = group_df[x_var].values.reshape(-1, 1)
        y = group_df[dv].values
        
        # Skip small groups
        if len(y) < min_size * 2:
            continue
            
        # Detect breakpoints
        algo = Binseg(model=model_type, min_size=min_size).fit(y)
        change_points = algo.predict(pen=pen)
        
        # Store results
        results[group_key] = {
            'change_points': change_points,
            'n_observations': len(group_df),
            'x_var': x_var,
            'dv': dv
        }
        
        # Visualization
        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, alpha=0.6, label='Data')
            
            # Plot breakpoints
            for cp in change_points[:-1]:
                plt.axvline(x=X[cp][0], color='red', linestyle='--', alpha=0.7)
            
            # Add trend line
            sns.regplot(x=X.flatten(), y=y, lowess=True, 
                        scatter=False, color='green', label='Trend')
            
            title = f"{dv} vs {x_var}"
            if group_vars:
                title += f" | Group: {group_key}"
            plt.title(title, fontsize=14)
            plt.xlabel(x_var, fontsize=12)
            plt.ylabel(dv, fontsize=12)
            plt.legend()
            plt.grid(alpha=0.2)
            plt.tight_layout()
    
    return results

def time_complexity_analysis(data, time_var='total_solve_time', size_var='items',
                             container=None, log_log=True, plot=True):
    """
    Analyzes time complexity scaling.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    time_var : str, optional
        Time variable (default: 'total_solve_time')
    size_var : str, optional
        Problem size variable (default: 'items')
    container : str, optional
        Filter for specific container type (default: None)
    log_log : bool, optional
        Use log-log scale (default: True)
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    dict: Regression results including coefficients
    """
    df = data.copy()
    
    # Filter by container if specified
    if container:
        df = df[df['container'] == container]
    
    # Prepare data
    X = df[size_var].values.reshape(-1, 1)
    y = df[time_var].values
    
    # Apply log transform if requested
    if log_log:
        X = np.log(X)
        y = np.log(y)
    
    # Add constant for regression
    X = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    
    # Extract coefficients
    results = {
        'intercept': model.params[0],
        'slope': model.params[1],
        'r_squared': model.rsquared,
        'p_value': model.f_pvalue,
        'model': model
    }
    
    # Visualization
    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 1], y, alpha=0.6, label='Data')
        
        # Plot regression line
        plt.plot(X[:, 1], model.predict(X), color='red', 
                 label=f'Fit: y = {results["slope"]:.2f}x + {results["intercept"]:.2f}')
        
        scale_type = "Log-Log" if log_log else "Linear"
        plt.title(f"{scale_type} Time Complexity Analysis", fontsize=14)
        xlabel = f"log({size_var})" if log_log else size_var
        ylabel = f"log({time_var})" if log_log else time_var
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
    
    return results

def constraint_impact_analysis(data, dv='packing_ratio', group_var='conservation',
                               test_type='mannwhitney', plot=True, palette='coolwarm'):
    """
    Analyzes impact of constraint types.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    dv : str, optional
        Dependent variable (default: 'packing_ratio')
    group_var : str, optional
        Grouping variable (default: 'conservation')
    test_type : str, optional
        Statistical test ('mannwhitney' or 'ttest') (default: 'mannwhitney')
    plot : bool, optional
        Generate visualization (default: True)
    palette : str, optional
        Color palette (default: 'coolwarm')
    
    Returns:
    tuple: Test statistic, p-value, effect size
    """
    df = data.copy()
    groups = df[group_var].unique()
    
    if len(groups) != 2:
        raise ValueError("This analysis requires exactly two groups")
    
    # Extract data for each group
    group1 = df[df[group_var] == groups[0]][dv]
    group2 = df[df[group_var] == groups[1]][dv]
    
    # Perform statistical test
    if test_type == 'mannwhitney':
        stat, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        # Calculate effect size (r = z / sqrt(n))
        z = stats.norm.ppf(1 - p/2)
        n = len(group1) + len(group2)
        effect_size = z / np.sqrt(n)
    elif test_type == 'ttest':
        stat, p = stats.ttest_ind(group1, group2, equal_var=False)
        # Cohen's d for effect size
        pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
        effect_size = (group1.mean() - group2.mean()) / pooled_std
    else:
        raise ValueError("Invalid test_type. Use 'mannwhitney' or 'ttest'")
    
    # Visualization
    if plot:
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=group_var, y=dv, data=df, inner='quartile', 
                       palette=palette, cut=0)
        sns.swarmplot(x=group_var, y=dv, data=df, color='black', alpha=0.5)
        plt.title(f"{dv} by {group_var}", fontsize=14)
        plt.xlabel(group_var, fontsize=12)
        plt.ylabel(dv, fontsize=12)
        plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
    
    return stat, p, effect_size

def container_constraint_interaction(data, dv='packing_ratio', 
                                    factors=['container', 'conservation'],
                                    plot=True, palette='viridis'):
    """
    Analyzes interaction between container and constraint types.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    dv : str, optional
        Dependent variable (default: 'packing_ratio')
    factors : list, optional
        Interaction factors (default: ['container', 'conservation'])
    plot : bool, optional
        Generate visualization (default: True)
    palette : str, optional
        Color palette (default: 'viridis')
    
    Returns:
    DataFrame: ANOVA results
    """
    # Create interaction term
    data['interaction'] = data[factors[0]] + " × " + data[factors[1]]
    
    # Fit ANOVA model
    formula = f"{dv} ~ C({factors[0]}) + C({factors[1]}) + C({factors[0]}):C({factors[1]})"
    model = smf.ols(formula, data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    # Visualization
    if plot:
        plt.figure(figsize=(12, 7))
        sns.pointplot(x=factors[0], y=dv, hue=factors[1], 
                      data=data, dodge=0.1, palette=palette,
                      markers=['o', 's', 'D'], linestyles=['-', '--', ':'])
        plt.title(f"Interaction Effect: {factors[0]} × {factors[1]}", fontsize=14)
        plt.ylabel(dv, fontsize=12)
        plt.xlabel(factors[0], fontsize=12)
        plt.legend(title=factors[1])
        plt.grid(alpha=0.2)
        plt.tight_layout()
    
    return anova_table

def cylinder_optimization_analysis(data, ratio_var='packing_ratio', 
                                   x_var='radius', y_var='height',
                                   softness_range=(0, 1), plot=True):
    """
    Optimizes cylinder aspect ratio.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset (cylinders only)
    ratio_var : str, optional
        Efficiency metric (default: 'packing_ratio')
    x_var : str, optional
        X-axis variable (default: 'radius')
    y_var : str, optional
        Y-axis variable (default: 'height')
    softness_range : tuple, optional
        Softness range to include (default: (0, 1))
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    DataFrame: Optimal configurations
    """
    # Filter cylinder data
    df = data[data['container'] == 'cylinder'].copy()
    df = df[(df['softness'] >= softness_range[0]) & 
            (df['softness'] <= softness_range[1])]
    
    # Calculate aspect ratio
    df['aspect_ratio'] = df[y_var] / df[x_var]
    
    # Find optimal aspect ratios
    optimal = df.groupby('softness').apply(
        lambda x: x.loc[x[ratio_var].idxmax()]
    ).reset_index(drop=True)
    
    # Visualization
    if plot:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x_var, y=y_var, size=ratio_var, hue=ratio_var,
                        data=df, palette='viridis', sizes=(20, 200))
        plt.scatter(optimal[x_var], optimal[y_var], s=100, marker='o', 
                    edgecolor='red', facecolor='none', label='Optimal')
        plt.title("Cylinder Optimization", fontsize=14)
        plt.xlabel(x_var, fontsize=12)
        plt.ylabel(y_var, fontsize=12)
        plt.legend(title=ratio_var)
        plt.grid(alpha=0.2)
        plt.tight_layout()
    
    return optimal

def solve_time_analysis(data, time_vars=['total_solve_time', 'ampl_time'], 
                        group_var='container', plot=True):
    """
    Analyzes solve time components.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    time_vars : list, optional
        Time components (default: ['total_solve_time', 'ampl_time'])
    group_var : str, optional
        Grouping variable (default: 'container')
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    DataFrame: Time analysis summary
    """
    df = data.copy()
    
    # Calculate time ratios
    df['ampl_ratio'] = df[time_vars[1]] / df[time_vars[0]]
    df['solve_ratio'] = 1 - df['ampl_ratio']
    
    # Summary statistics
    summary = df.groupby(group_var)[['ampl_ratio', 'solve_ratio']].describe()
    
    # Visualization
    if plot:
        melt_df = df.melt(id_vars=[group_var], 
                          value_vars=['ampl_ratio', 'solve_ratio'],
                          var_name='time_component', value_name='ratio')
        
        plt.figure(figsize=(12, 7))
        sns.boxplot(x=group_var, y='ratio', hue='time_component', 
                    data=melt_df, palette='Set2')
        plt.title("Time Component Distribution", fontsize=14)
        plt.ylabel("Time Ratio", fontsize=12)
        plt.xlabel(group_var, fontsize=12)
        plt.legend(title="Component")
        plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
    
    return summary

def packing_ratio_model(data, dv='packing_ratio', predictors=['container', 
                        'softness', 'items', 'conservation'],
                        model_type='beta', plot=True):
    """
    Models packing ratio using advanced regression.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    dv : str, optional
        Dependent variable (default: 'packing_ratio')
    predictors : list, optional
        Predictor variables (default: ['container', 'softness', 
                                      'items', 'conservation'])
    model_type : str, optional
        Model type ('beta' or 'ols') (default: 'beta')
    plot : bool, optional
        Generate diagnostic plots (default: True)
    
    Returns:
    Regression model object
    """
    formula = f"{dv} ~ " + " + ".join([f"C({p})" if data[p].dtype == 'O' else p 
                                      for p in predictors])
    
    if model_type == 'beta':
        # Beta regression for bounded [0,1] outcomes
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod.families import Binomial
        family = Binomial()
        model = GLM.from_formula(formula, data, family=family).fit()
    else:
        # Standard OLS
        model = smf.ols(formula, data=data).fit()
    
    # Diagnostic plots
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residuals vs Fitted
        sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=axes[0])
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_title('Residuals vs Fitted')
        axes[0].set_xlabel('Fitted values')
        axes[0].set_ylabel('Residuals')
        
        # QQ Plot
        stats.probplot(model.resid, dist="norm", plot=axes[1])
        axes[1].set_title('Normal Q-Q Plot')
        
        plt.tight_layout()
    
    return model

def feasibility_analysis(data, outcome_var='solve_result', predictors=['softness', 
                         'items', 'container'], plot=True):
    """
    Analyzes solution feasibility.
    
    Parameters:
    data : DataFrame
        Full dataset including infeasible solutions
    outcome_var : str, optional
        Outcome variable (default: 'solve_result')
    predictors : list, optional
        Predictor variables (default: ['softness', 'items', 'container'])
    plot : bool, optional
        Generate ROC curve (default: True)
    
    Returns:
    LogisticRegression model
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import LabelEncoder
    
    df = data.copy()
    
    # Encode outcome
    le = LabelEncoder()
    y = le.fit_transform(df[outcome_var])
    
    # Prepare features
    X = pd.get_dummies(df[predictors], drop_first=True)
    
    # Fit logistic regression
    model = LogisticRegression(penalty='l2', max_iter=1000)
    model.fit(X, y)
    
    # Generate predictions
    y_pred = model.predict_proba(X)[:, 1]
    
    # ROC curve
    if plot:
        fpr, tpr, _ = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.2)
    
    return model

def container_space_visualization(data, plot_type='3d', color_var='packing_ratio', 
                                  size_var='items', symbol_var='container'):
    """
    Interactive container space visualization.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    plot_type : str, optional
        Visualization type ('3d' or 'scatter_matrix') (default: '3d')
    color_var : str, optional
        Color mapping variable (default: 'packing_ratio')
    size_var : str, optional
        Size mapping variable (default: 'items')
    symbol_var : str, optional
        Symbol mapping variable (default: 'container')
    
    Returns:
    Plotly figure object
    """
    df = data.copy()
    
    if plot_type == '3d':
        fig = px.scatter_3d(
            df, 
            x='radius' if 'radius' in df else 'side',
            y='height' if 'height' in df else 'side',
            z=color_var,
            color=color_var,
            size=size_var,
            symbol=symbol_var,
            hover_name='job_id',
            opacity=0.7,
            title="Container Space Visualization"
        )
    else:
        dimensions = ['radius', 'height', 'side', 'packing_ratio', 'softness']
        dimensions = [d for d in dimensions if d in df.columns]
        fig = px.scatter_matrix(
            df,
            dimensions=dimensions,
            color='container',
            symbol='conservation',
            hover_name='job_id',
            title="Container Space Scatter Matrix"
        )
    
    return fig


def normality_assessment(data, variables=['packing_ratio', 'total_solve_time'], 
                         alpha=0.05, plot=True):
    """
    Assesses normality of specified variables using statistical tests and visualizations.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    variables : list, optional
        Variables to assess (default: ['packing_ratio', 'total_solve_time'])
    alpha : float, optional
        Significance level (default: 0.05)
    plot : bool, optional
        Generate visualizations (default: True)
    
    Returns:
    dict: Normality test results for each variable
    """
    results = {}
    
    for var in variables:
        # Remove missing values
        clean_data = data[var].dropna()
        
        # Shapiro-Wilk test
        stat, p_value = shapiro(clean_data)
        is_normal = p_value > alpha
        
        # Store results
        results[var] = {
            'shapiro_stat': stat,
            'p_value': p_value,
            'is_normal': is_normal,
            'n': len(clean_data)
        }
        
        # Visualization
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram with KDE
            sns.histplot(clean_data, kde=True, ax=axes[0])
            axes[0].set_title(f'{var} Distribution')
            axes[0].set_xlabel(var)
            
            # Q-Q Plot
            probplot(clean_data, dist='norm', plot=axes[1])
            axes[1].set_title(f'Q-Q Plot for {var}')
            
            plt.tight_layout()
    
    return results

def heteroscedasticity_check(data, dv='packing_ratio', predictors=['softness', 'items'], 
                             alpha=0.05, plot=True):
    """
    Checks for heteroscedasticity in regression models.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    dv : str, optional
        Dependent variable (default: 'packing_ratio')
    predictors : list, optional
        Independent variables (default: ['softness', 'items'])
    alpha : float, optional
        Significance level (default: 0.05)
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    dict: Test results including p-value and conclusion
    """
    # Prepare data
    X = data[predictors]
    y = data[dv]
    
    # Add constant for OLS
    X = sm.add_constant(X)
    
    # Fit OLS model
    model = sm.OLS(y, X).fit()
    
    # Perform Breusch-Pagan test
    bp_test = het_breuschpagan(model.resid, X)
    
    # Interpret results
    p_value = bp_test[1]
    is_homoscedastic = p_value > alpha
    
    results = {
        'bp_statistic': bp_test[0],
        'p_value': p_value,
        'is_homoscedastic': is_homoscedastic,
        'test': 'Breusch-Pagan'
    }
    
    # Visualization
    if plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(model.fittedvalues, model.resid, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals vs Fitted Values', fontsize=14)
        plt.xlabel('Fitted Values', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.grid(alpha=0.2)
        
        # Add heteroscedasticity indicator
        plt.annotate(f"Heteroscedasticity: {'Present' if not is_homoscedastic else 'Absent'}",
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    return results

def time_per_tetra_analysis(data, time_var='total_solve_time', size_var='items',
                            group_var='container', plot=True, log_scale=True):
    """
    Analyzes time per tetrahedron across container types.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    time_var : str, optional
        Time variable (default: 'total_solve_time')
    size_var : str, optional
        Problem size variable (default: 'items')
    group_var : str, optional
        Grouping variable (default: 'container')
    plot : bool, optional
        Generate visualization (default: True)
    log_scale : bool, optional
        Use log scale for y-axis (default: True)
    
    Returns:
    DataFrame: Summary statistics by group
    """
    # Calculate time per tetrahedron
    data['time_per_tetra'] = data[time_var] / data[size_var]
    
    # Group summary
    summary = data.groupby(group_var)['time_per_tetra'].describe()
    
    # Visualization
    if plot:
        plt.figure(figsize=(12, 7))
        sns.boxplot(x=group_var, y='time_per_tetra', data=data, showfliers=False)
        sns.stripplot(x=group_var, y='time_per_tetra', data=data, 
                      color='black', alpha=0.3, jitter=True)
        
        if log_scale:
            plt.yscale('log')
            plt.ylabel('log(Time per Tetrahedron)')
        else:
            plt.ylabel('Time per Tetrahedron')
            
        plt.title('Time per Tetrahedron by Container Type', fontsize=14)
        plt.xlabel('Container Type', fontsize=12)
        plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
    
    return summary

def success_rate_analysis(data, outcome_var='solve_result', group_vars=['container', 'conservation'],
                          plot=True, palette='viridis'):
    """
    Analyzes success rates across experimental conditions.
    
    Parameters:
    data : DataFrame
        Full dataset including infeasible solutions
    outcome_var : str, optional
        Outcome variable (default: 'solve_result')
    group_vars : list, optional
        Grouping variables (default: ['container', 'conservation'])
    plot : bool, optional
        Generate visualization (default: True)
    palette : str, optional
        Color palette (default: 'viridis')
    
    Returns:
    DataFrame: Success rates by group
    """
    # Calculate success rates
    data['success'] = data[outcome_var].apply(lambda x: 1 if x == 'solved' else 0)
    grouped = data.groupby(group_vars)['success'].agg(['mean', 'count', 'std'])
    grouped.columns = ['success_rate', 'n_observations', 'std_dev']
    
    # Calculate confidence intervals
    grouped['ci_low'] = grouped['success_rate'] - 1.96 * grouped['std_dev'] / np.sqrt(grouped['n_observations'])
    grouped['ci_high'] = grouped['success_rate'] + 1.96 * grouped['std_dev'] / np.sqrt(grouped['n_observations'])
    
    # Visualization
    if plot:
        grouped = grouped.reset_index()
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=group_vars[0], y='success_rate', hue=group_vars[1],
                         data=grouped, palette=palette)
        
        # Add error bars
        for i, bar in enumerate(ax.patches):
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            ci_low = grouped.iloc[i]['ci_low']
            ci_high = grouped.iloc[i]['ci_high']
            plt.errorbar(x, y, yerr=[[y - ci_low], [ci_high - y]], 
                         fmt='none', c='black', capsize=5)
        
        plt.title('Success Rate by Experimental Conditions', fontsize=14)
        plt.ylabel('Success Rate', fontsize=12)
        plt.xlabel(group_vars[0], fontsize=12)
        plt.ylim(0, 1.1)
        plt.legend(title=group_vars[1], loc='upper right')
        plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
    
    return grouped

def dimensional_pca(data, variables=['radius', 'height', 'side', 'container_volume'], 
                   group_var='container', n_components=2, plot=True):
    """
    Performs PCA on dimensional variables to reduce dimensionality.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    variables : list, optional
        Variables for PCA (default: ['radius', 'height', 'side', 'container_volume'])
    group_var : str, optional
        Grouping variable for coloring (default: 'container')
    n_components : int, optional
        Number of principal components (default: 2)
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    tuple: (PCA model, transformed data, variance explained)
    """
    # Prepare data - only include specified variables and drop missing
    df = data[variables + [group_var]].dropna()
    X = df[variables]
    groups = df[group_var]
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)
    
    # Create results DataFrame
    pc_cols = [f'PC{i+1}' for i in range(n_components)]
    results_df = pd.DataFrame(data=principal_components, columns=pc_cols)
    results_df[group_var] = groups.values
    
    # Variance explained
    variance_explained = pca.explained_variance_ratio_
    
    # Visualization
    if plot:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x=pc_cols[0], y=pc_cols[1], hue=group_var, 
                        data=results_df, palette='viridis', s=100)
        
        # Add variable vectors
        for i, var in enumerate(variables):
            plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
                      color='r', alpha=0.7, width=0.01)
            plt.text(pca.components_[0, i]*1.15, pca.components_[1, i]*1.15, 
                     var, color='r', fontsize=12)
        
        plt.title(f'PCA of Container Dimensions (Variance: {variance_explained.sum():.1%})', fontsize=14)
        plt.xlabel(f'PC1 ({variance_explained[0]:.1%})', fontsize=12)
        plt.ylabel(f'PC2 ({variance_explained[1]:.1%})', fontsize=12)
        plt.grid(alpha=0.2)
        plt.legend(title=group_var)
        plt.tight_layout()
    
    return pca, results_df, variance_explained

def softness_binning_analysis(data, dv='packing_ratio', x_var='softness', 
                              group_var='container', bins=5, plot=True):
    """
    Analyzes packing ratio across softness bins.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    dv : str, optional
        Dependent variable (default: 'packing_ratio')
    x_var : str, optional
        Variable to bin (default: 'softness')
    group_var : str, optional
        Grouping variable (default: 'container')
    bins : int or list, optional
        Number of bins or bin edges (default: 5)
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    DataFrame: Summary statistics by softness bin
    """
    # Create softness bins
    if isinstance(bins, int):
        data['softness_bin'] = pd.qcut(data[x_var], bins, duplicates='drop')
    else:
        data['softness_bin'] = pd.cut(data[x_var], bins)
    
    # Group summary
    grouped = data.groupby(['softness_bin', group_var])[dv].agg(['mean', 'std', 'count'])
    grouped.columns = ['mean', 'std_dev', 'n_observations']
    grouped = grouped.reset_index()
    
    # Calculate confidence intervals
    grouped['ci'] = 1.96 * grouped['std_dev'] / np.sqrt(grouped['n_observations'])
    
    # Visualization
    if plot:
        plt.figure(figsize=(12, 7))
        sns.pointplot(x='softness_bin', y='mean', hue=group_var,
                      data=grouped, dodge=0.2, palette='viridis',
                      markers=['o', 's', 'D'], linestyles=['-', '--', ':'])
        
        # Add error bars
        for i, row in grouped.iterrows():
            plt.errorbar(i % len(grouped['softness_bin'].unique()), row['mean'], 
                         yerr=row['ci'], fmt='none', c='black', capsize=5)
        
        plt.title(f'{dv} by Softness Bins', fontsize=14)
        plt.ylabel(f'Mean {dv}', fontsize=12)
        plt.xlabel('Softness Bin', fontsize=12)
        plt.legend(title=group_var)
        plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
    
    return grouped

def volume_efficiency_analysis(data, eff_var='packing_ratio', size_var='container_volume',
                               group_var='container', plot=True, log_log=False):
    """
    Analyzes efficiency vs container volume.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    eff_var : str, optional
        Efficiency variable (default: 'packing_ratio')
    size_var : str, optional
        Size variable (default: 'container_volume')
    group_var : str, optional
        Grouping variable (default: 'container')
    plot : bool, optional
        Generate visualization (default: True)
    log_log : bool, optional
        Use log-log scale (default: False)
    
    Returns:
    dict: Correlation results by group
    """
    # Calculate correlations by group
    groups = data[group_var].unique()
    results = {}
    
    for group in groups:
        group_data = data[data[group_var] == group]
        
        if log_log:
            x = np.log(group_data[size_var])
            y = np.log(group_data[eff_var])
        else:
            x = group_data[size_var]
            y = group_data[eff_var]
        
        # Remove missing/infinite values
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        
        # Calculate correlation
        corr, p_value = stats.pearsonr(x, y)
        
        results[group] = {
            'correlation': corr,
            'p_value': p_value,
            'n': len(x)
        }
    
    # Visualization
    if plot:
        plt.figure(figsize=(12, 7))
        for group in groups:
            group_data = data[data[group_var] == group]
            
            if log_log:
                x = np.log(group_data[size_var])
                y = np.log(group_data[eff_var])
                xlabel = f'log({size_var})'
                ylabel = f'log({eff_var})'
            else:
                x = group_data[size_var]
                y = group_data[eff_var]
                xlabel = size_var
                ylabel = eff_var
            
            # Remove missing/infinite values
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            
            # Scatter plot with regression line
            sns.regplot(x=x, y=y, label=group, scatter_kws={'alpha': 0.6})
        
        plt.title(f'{eff_var} vs {size_var}', fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(title=group_var)
        plt.grid(alpha=0.2)
        plt.tight_layout()
    
    return results

def time_distribution_analysis(data, time_var='total_solve_time', 
                               group_var='container', plot=True, log_scale=True):
    """
    Analyzes the distribution of solve times.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    time_var : str, optional
        Time variable (default: 'total_solve_time')
    group_var : str, optional
        Grouping variable (default: 'container')
    plot : bool, optional
        Generate visualization (default: True)
    log_scale : bool, optional
        Use log scale for time (default: True)
    
    Returns:
    DataFrame: Summary statistics by group
    """
    # Summary statistics
    summary = data.groupby(group_var)[time_var].describe()
    
    # Visualization
    if plot:
        plt.figure(figsize=(12, 7))
        
        if log_scale:
            # Use KDE plots for better visualization on log scale
            for group in data[group_var].unique():
                group_data = data[data[group_var] == group][time_var]
                log_data = np.log(group_data[group_data > 0])
                
                # Kernel Density Estimation
                kde = gaussian_kde(log_data)
                x_vals = np.linspace(log_data.min(), log_data.max(), 1000)
                plt.plot(x_vals, kde(x_vals), label=group)
            
            plt.xlabel(f'log({time_var})')
            plt.ylabel('Density')
            plt.title(f'Distribution of log({time_var}) by {group_var}', fontsize=14)
        else:
            # Boxplot for linear scale
            sns.boxplot(x=group_var, y=time_var, data=data, showfliers=False)
            sns.stripplot(x=group_var, y=time_var, data=data, color='black', alpha=0.3, jitter=True)
            plt.ylabel(time_var)
            plt.title(f'Distribution of {time_var} by {group_var}', fontsize=14)
        
        plt.xlabel(group_var, fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
    
    return summary

def multivariate_outliers(data, variables=['packing_ratio', 'total_solve_time', 'softness'],
                          threshold=3, plot=True):
    """
    Identifies multivariate outliers using Mahalanobis distance.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    variables : list, optional
        Variables to include (default: ['packing_ratio', 'total_solve_time', 'softness'])
    threshold : float, optional
        Z-score threshold for outliers (default: 3)
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    DataFrame: Data with outlier flags
    """
    # Prepare data
    df = data[variables].dropna()
    
    # Calculate Mahalanobis distance
    cov = np.cov(df.values, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    mean = np.mean(df.values, axis=0)
    
    mahalanobis = []
    for i, row in df.iterrows():
        diff = row.values - mean
        distance = np.sqrt(diff.dot(inv_cov).dot(diff.T))
        mahalanobis.append(distance)
    
    # Calculate z-scores
    z_scores = zscore(mahalanobis)
    is_outlier = np.abs(z_scores) > threshold
    
    # Create results DataFrame
    results = data.copy()
    results['mahalanobis'] = np.nan
    results['mahalanobis_z'] = np.nan
    results['is_outlier'] = False
    
    results.loc[df.index, 'mahalanobis'] = mahalanobis
    results.loc[df.index, 'mahalanobis_z'] = z_scores
    results.loc[df.index, 'is_outlier'] = is_outlier
    
    # Visualization
    if plot:
        plt.figure(figsize=(12, 6))
        
        # Mahalanobis distribution
        plt.subplot(1, 2, 1)
        sns.histplot(mahalanobis, kde=True)
        plt.axvline(np.mean(mahalanobis), color='r', linestyle='--')
        plt.title('Mahalanobis Distance Distribution')
        plt.xlabel('Mahalanobis Distance')
        
        # Z-scores
        plt.subplot(1, 2, 2)
        sns.scatterplot(x=range(len(z_scores)), y=z_scores, hue=is_outlier)
        plt.axhline(threshold, color='r', linestyle='--')
        plt.axhline(-threshold, color='r', linestyle='--')
        plt.title('Mahalanobis Z-Scores')
        plt.xlabel('Observation Index')
        plt.ylabel('Z-Score')
        plt.legend(title='Outlier')
        
        plt.tight_layout()
    
    return results

def optimal_aspect_ratio(data, ratio_var='packing_ratio', 
                         x_var='radius', y_var='height',
                         softness_range=(0, 1), plot=True):
    """
    Identifies optimal aspect ratios for cylinders.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    ratio_var : str, optional
        Efficiency metric (default: 'packing_ratio')
    x_var : str, optional
        X-axis variable (default: 'radius')
    y_var : str, optional
        Y-axis variable (default: 'height')
    softness_range : tuple, optional
        Softness range to include (default: (0, 1))
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    DataFrame: Optimal configurations
    """
    # Filter cylinder data
    df = data[data['container'] == 'cylinder'].copy()
    df = df[(df['softness'] >= softness_range[0]) & 
            (df['softness'] <= softness_range[1])]
    
    # Calculate aspect ratio
    df['aspect_ratio'] = df[y_var] / df[x_var]
    
    # Find optimal aspect ratios
    optimal = df.groupby('softness').apply(
        lambda x: x.loc[x[ratio_var].idxmax()]
    ).reset_index(drop=True)
    
    # Visualization
    if plot:
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        ax = sns.scatterplot(x=x_var, y=y_var, size=ratio_var, hue=ratio_var,
                             data=df, palette='viridis', sizes=(20, 200))
        
        # Optimal points
        plt.scatter(optimal[x_var], optimal[y_var], s=100, marker='o', 
                    edgecolor='red', facecolor='none', label='Optimal')
        
        # Connect optimal points
        plt.plot(optimal[x_var], optimal[y_var], 'r--', alpha=0.5)
        
        # Add aspect ratio labels
        for i, row in optimal.iterrows():
            plt.annotate(f"AR: {row['aspect_ratio']:.2f}", 
                         (row[x_var], row[y_var]),
                         xytext=(10, -10), textcoords='offset points')
        
        plt.title('Cylinder Optimization', fontsize=14)
        plt.xlabel(x_var, fontsize=12)
        plt.ylabel(y_var, fontsize=12)
        plt.legend(title=ratio_var)
        plt.grid(alpha=0.2)
        plt.tight_layout()
    
    return optimal

def constraint_facet_analysis(data, dv='packing_ratio', x_var='softness',
                              row_var='container', col_var='conservation',
                              plot=True):
    """
    Creates faceted plots for constraint analysis.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    dv : str, optional
        Dependent variable (default: 'packing_ratio')
    x_var : str, optional
        X-axis variable (default: 'softness')
    row_var : str, optional
        Row facet variable (default: 'container')
    col_var : str, optional
        Column facet variable (default: 'conservation')
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    None (displays plot)
    """
    if plot:
        g = sns.FacetGrid(data, row=row_var, col=col_var, 
                          margin_titles=True, height=4, aspect=1.2)
        g.map_dataframe(sns.scatterplot, x=x_var, y=dv, alpha=0.7)
        g.map_dataframe(sns.regplot, x=x_var, y=dv, scatter=False, 
                        line_kws={'color': 'red', 'alpha': 0.7})
        g.set_axis_labels(x_var, dv)
        g.fig.suptitle(f'{dv} by {x_var} across Constraints', y=1.03)
        plt.tight_layout()

def spatial_efficiency_analysis(data, eff_var='packing_ratio', 
                                x_var='radius', y_var='height',
                                group_var='container', plot=True):
    """
    Analyzes spatial efficiency using hexbin plots.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    eff_var : str, optional
        Efficiency metric (default: 'packing_ratio')
    x_var : str, optional
        X-axis variable (default: 'radius')
    y_var : str, optional
        Y-axis variable (default: 'height')
    group_var : str, optional
        Grouping variable (default: 'container')
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    None (displays plot)
    """
    if plot:
        g = sns.FacetGrid(data, col=group_var, col_wrap=3, height=5)
        g.map_dataframe(lambda data, color: plt.hexbin(
            data[x_var], data[y_var], C=data[eff_var], 
            gridsize=20, cmap='viridis', reduce_C_function=np.mean
        ))
        g.set_axis_labels(x_var, y_var)
        g.fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=g.axes)
        g.fig.suptitle(f'Spatial Efficiency of {eff_var}', y=1.03)
        plt.tight_layout()

def statistical_power_analysis(data, dv='packing_ratio', group_var='container',
                               effect_size=0.5, alpha=0.05, plot=True):
    """
    Estimates statistical power for group comparisons.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    dv : str, optional
        Dependent variable (default: 'packing_ratio')
    group_var : str, optional
        Grouping variable (default: 'container')
    effect_size : float, optional
        Minimum detectable effect size (default: 0.5)
    alpha : float, optional
        Significance level (default: 0.05)
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    dict: Power analysis results
    """
    from statsmodels.stats.power import FTestAnovaPower
    
    # Prepare data
    groups = data.groupby(group_var)[dv].apply(list)
    n_groups = len(groups)
    group_sizes = [len(g) for g in groups]
    
    # Estimate power
    power_analysis = FTestAnovaPower()
    power = power_analysis.power(effect_size, np.mean(group_sizes), n_groups, alpha)
    
    results = {
        'effect_size': effect_size,
        'alpha': alpha,
        'n_groups': n_groups,
        'avg_group_size': np.mean(group_sizes),
        'power': power
    }
    
    # Visualization
    if plot:
        # Create power curve
        effect_sizes = np.linspace(0.1, 1.0, 50)
        powers = power_analysis.power(effect_sizes, np.mean(group_sizes), n_groups, alpha)
        
        plt.figure(figsize=(10, 6))
        plt.plot(effect_sizes, powers, 'b-', linewidth=2)
        plt.axhline(0.8, color='r', linestyle='--', label='80% Power')
        plt.axvline(effect_size, color='g', linestyle='--', label='Effect Size')
        plt.title('Statistical Power Analysis', fontsize=14)
        plt.xlabel('Effect Size (Cohen\'s f)')
        plt.ylabel('Statistical Power')
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
    
    return results

def cluster_analysis(data, variables=['packing_ratio', 'total_solve_time', 'softness'],
                     n_clusters=3, plot=True):
    """
    Performs cluster analysis to identify similar configurations.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    variables : list, optional
        Variables for clustering (default: ['packing_ratio', 'total_solve_time', 'softness'])
    n_clusters : int, optional
        Number of clusters (default: 3)
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    tuple: (Cluster labels, cluster centers, silhouette score)
    """
    # Prepare data
    df = data[variables].dropna()
    X = df.values
    
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    
    # Get cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Add cluster labels to data
    results = data.copy()
    results['cluster'] = np.nan
    results.loc[df.index, 'cluster'] = cluster_labels
    
    # Visualization
    if plot:
        # 2D visualization (first two variables)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], 
                    marker='X', s=200, c='red', label='Cluster Centers')
        plt.title(f'Cluster Analysis (Silhouette: {silhouette_avg:.2f})', fontsize=14)
        plt.xlabel(variables[0], fontsize=12)
        plt.ylabel(variables[1], fontsize=12)
        plt.legend()
        plt.grid(alpha=0.2)
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
    
    return cluster_labels, cluster_centers, silhouette_avg

def time_packing_tradeoff(data, time_var='total_solve_time', eff_var='packing_ratio',
                          group_var='container', plot=True):
    """
    Analyzes the tradeoff between solve time and packing efficiency.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    time_var : str, optional
        Time variable (default: 'total_solve_time')
    eff_var : str, optional
        Efficiency variable (default: 'packing_ratio')
    group_var : str, optional
        Grouping variable (default: 'container')
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    dict: Correlation results by group
    """
    # Calculate correlations by group
    groups = data[group_var].unique()
    results = {}
    
    for group in groups:
        group_data = data[data[group_var] == group]
        
        # Calculate correlation
        corr, p_value = stats.spearmanr(group_data[time_var], group_data[eff_var])
        
        results[group] = {
            'correlation': corr,
            'p_value': p_value,
            'n': len(group_data)
        }
    
    # Visualization
    if plot:
        plt.figure(figsize=(12, 7))
        for group in groups:
            group_data = data[data[group_var] == group]
            plt.scatter(group_data[time_var], group_data[eff_var], 
                        alpha=0.6, label=group)
        
        plt.title('Time vs Packing Efficiency Tradeoff', fontsize=14)
        plt.xlabel(time_var, fontsize=12)
        plt.ylabel(eff_var, fontsize=12)
        plt.legend(title=group_var)
        plt.grid(alpha=0.2)
        
        # Add correlation annotations
        for group, res in results.items():
            plt.annotate(f"{group}: r={res['correlation']:.2f}", 
                         xy=(0.05, 0.85 - list(results.keys()).index(group)*0.05),
                         xycoords='axes fraction')
        
        plt.tight_layout()
    
    return results

def interactive_softness_analysis(data, eff_var='packing_ratio', time_var='total_solve_time',
                                  group_var='container'):
    """
    Creates an interactive visualization of softness effects.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    eff_var : str, optional
        Efficiency variable (default: 'packing_ratio')
    time_var : str, optional
        Time variable (default: 'total_solve_time')
    group_var : str, optional
        Grouping variable (default: 'container')
    
    Returns:
    Plotly figure
    """
    fig = px.scatter(data, x='softness', y=eff_var, 
                     size=time_var, color=group_var,
                     hover_name='job_id', 
                     animation_frame='conservation',
                     title='Softness Effects on Packing Efficiency')
    
    return fig

def bootstrap_validation(data, dv='packing_ratio', predictors=['softness', 'items'],
                         n_bootstraps=1000, alpha=0.05, plot=True):
    """
    Performs bootstrap validation for regression models.
    
    Parameters:
    data : DataFrame
        Preprocessed dataset
    dv : str, optional
        Dependent variable (default: 'packing_ratio')
    predictors : list, optional
        Predictor variables (default: ['softness', 'items'])
    n_bootstraps : int, optional
        Number of bootstrap samples (default: 1000)
    alpha : float, optional
        Significance level for confidence intervals (default: 0.05)
    plot : bool, optional
        Generate visualization (default: True)
    
    Returns:
    DataFrame: Bootstrap results with confidence intervals
    """
    # Prepare data
    X = data[predictors]
    y = data[dv]
    
    # Add constant
    X = sm.add_constant(X)
    
    # Original model
    orig_model = sm.OLS(y, X).fit()
    
    # Bootstrap coefficients
    boot_coefs = np.zeros((n_bootstraps, len(orig_model.params)))
    
    for i in range(n_bootstraps):
        # Resample with replacement
        indices = np.random.choice(len(y), len(y), replace=True)
        X_boot = X.iloc[indices]
        y_boot = y.iloc[indices]
        
        # Fit model
        try:
            model = sm.OLS(y_boot, X_boot).fit()
            boot_coefs[i] = model.params
        except:
            boot_coefs[i] = np.nan
    
    # Calculate confidence intervals
    ci_lower = np.nanpercentile(boot_coefs, 100*alpha/2, axis=0)
    ci_upper = np.nanpercentile(boot_coefs, 100*(1-alpha/2), axis=0)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'variable': orig_model.params.index,
        'coef': orig_model.params,
        'std_err': orig_model.bse,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })
    
    # Visualization
    if plot:
        plt.figure(figsize=(12, 6))
        
        # Create error bars
        y_pos = np.arange(len(results))
        plt.errorbar(results['coef'], y_pos, 
                     xerr=[results['coef'] - results['ci_lower'], 
                           results['ci_upper'] - results['coef']],
                     fmt='o', capsize=5)
        
        plt.yticks(y_pos, results['variable'])
        plt.axvline(0, color='r', linestyle='--')
        plt.title('Bootstrap Confidence Intervals for Coefficients', fontsize=14)
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.grid(alpha=0.2)
        plt.tight_layout()
    
    return results


df = pd.read_excel('artifacts/export-20250714.xlsx')
df = df[
    (df['valid_result'] == True) &
    (df['result'] == 'solved')
]
df = df.drop(
    [
        'solver',
        'valid_result'
    ],
    axis=1,
    errors='ignore'
)



# container_efficiency = container_efficiency_analysis(df)
# softness_thresholds = softness_threshold_detection(df, group_vars=['container'])
time_complexity = time_complexity_analysis(df)
# constraint_impact = constraint_impact_analysis(df)
interaction_effects = container_constraint_interaction(df)
cylinder_optima = cylinder_optimization_analysis(df)
time_components = solve_time_analysis(df)
packing_model = packing_ratio_model(df)
feasibility_model = feasibility_analysis(pd.read_excel("export-20250711.xlsx"))  # Full data
container_3d = container_space_visualization(df)
normality = normality_assessment(df)
heteroscedasticity = heteroscedasticity_check(df)
time_per_tetra = time_per_tetra_analysis(df)
success_rates = success_rate_analysis(pd.read_excel("export-20250711.xlsx"))
pca_results = dimensional_pca(df)
softness_bins = softness_binning_analysis(df)
volume_efficiency = volume_efficiency_analysis(df)
time_distribution = time_distribution_analysis(df)
outliers = multivariate_outliers(df)
constraint_facets = constraint_facet_analysis(df)
spatial_efficiency = spatial_efficiency_analysis(df)
statistical_power = statistical_power_analysis(df)
clusters = cluster_analysis(df)
tradeoff = time_packing_tradeoff(df)
interactive_plot = interactive_softness_analysis(df)
bootstrap_results = bootstrap_validation(df)

# 4. Save or display results
plt.show()  # For matplotlib plots
container_3d.show()  # For Plotly figures