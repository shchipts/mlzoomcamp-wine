import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def show_histograms(df):
    fig = plt.figure(figsize=(16, 18))

    n_cols = 3
    n_super_rows = int(df.shape[1] / n_cols)
    n_rows = n_super_rows * 3
    
    color = sns.color_palette('deep')[0]
    kde_color = tuple(c * 1.5 for c in to_rgb(color))
    
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        hspace=0.29,
        wspace=0.2,
        height_ratios=[4, 0.5, 0.5] * n_super_rows,
    )
    
    for i, col in enumerate(df.columns):
        row = (i // n_cols) * 3
        column = i % n_cols
    
        ax_hist = fig.add_subplot(gs[row, column])
        ax_hist.set_axisbelow(True)
        ax_hist.grid(False)
        ax_hist.spines['top'].set_visible(False)
        ax_hist.spines['right'].set_visible(False)
        ax_hist.spines['left'].set_color('black')
        ax_hist.spines['bottom'].set_color('black')
        sns.histplot(
            df[col],
            kde=True,
            ax=ax_hist,
            color=color,
            alpha=1,
            line_kws={'color': kde_color, 'linewidth': 2},
        )
        ax_hist.set_title(col, fontsize=18, fontweight='bold')
        ax_hist.set_xlabel('')
        ax_hist.tick_params(axis='x', labelsize=12)
    
        ax_box = fig.add_subplot(gs[row + 1, column])
        ax_box.axis('off')
        sns.boxplot(
            x=df[col],
            ax=ax_box,
            color=color,
            saturation=1,
            boxprops={'edgecolor': 'black'},
            whiskerprops={'color': 'black'},
            capprops={'color': 'black'},
            medianprops={'color': 'black', 'linewidth': 1.5},
            flierprops={
                'markeredgecolor': 'black',
                'markerfacecolor': 'black',
                'markersize': 4,
            },
        )
        ax_box.set_xlabel('')
    
    plt.show()

def show_correlation_matrix(df):

    # plot correlation matrix
    column_order = [col for col in df.columns if col != 'quality'] + ['quality']

    plt.figure(figsize=(6.4, 4.8))
    plt.matshow(df[column_order].corr(), vmin=-1, vmax=1, fignum=1)
    cb = plt.colorbar(pad=0.15, shrink=0.65)
    cb.ax.tick_params(labelsize=10)
    plt.xticks(range(len(column_order)), column_order, rotation=60, ha='left')
    plt.yticks(range(len(column_order)), column_order)
    plt.title('Correlation Matrix', fontsize=16, fontweight='bold')
    
    plt.show()

def correlation_analysis(df):
    feature_corr = df.drop(columns=['quality']).corr()

    upper_triangle = feature_corr.where(
        np.triu(np.ones(feature_corr.shape), k=1).astype(bool)
    )    
    correlations = upper_triangle.stack().reset_index()    
    correlations.columns = ['Feature 1', 'Feature 2', 'Correlation']
    
    bins = [0, 0.3, 0.5, 0.7, 0.9, 1]
    names = ['Weak', 'Moderate', 'Strong', 'Very strong', 'Extremely strong']
    correlations['Interpretation'] = pd.cut(
        abs(correlations['Correlation']), bins, labels=names
    )
    
    correlations.sort_values(by=['Interpretation'], ascending=False, ignore_index=True)

    return correlations

def show_VIFs(df):
    X = df.drop(columns=['quality'])
    X_const = add_constant(X)
    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_const.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])
    ]
    vif_data = vif_data[vif_data['Feature'] != 'const']
    vif_data = vif_data.sort_values(by='VIF', ascending=False)
    

    plt.figure()

    sns.barplot(data=vif_data, x='VIF', y='Feature', color='#1f77b4', legend=False)

    for i, value in enumerate(vif_data['VIF']):
        plt.text(value + 0.1, i, f'{value:.2f}', va='center', fontsize=9)
    
    plt.axvline(x=5, color='#ff7f0e', label='VIF ≥ 5: potentially problematic')
    plt.axvline(x=10, color='#a64b00', label='VIF ≥ 10: serious multicollinearity concern')
    
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(range(0, 11, 1))
    plt.xlim(0, max(11, vif_data['VIF'].max() + 1))
    
    plt.title('Variance Inflation Factor (VIF)', pad=50, fontsize=16, fontweight='bold')
    
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=1, frameon=False)
    
    ax = plt.gca()
    
    ax.set_axisbelow(True)
    ax.grid(False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    plt.tight_layout()
    plt.show()