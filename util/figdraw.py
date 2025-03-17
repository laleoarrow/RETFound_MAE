import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

# Set publication-quality plot parameters for general plots
def set_basic_style():
    """Set the plotting style for standard visualizations."""
    # Reset to default style first
    plt.rcdefaults()
    
    # Apply academic/scientific style settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Liberation Sans', 'sans-serif'],
        'font.size': 9,
        'axes.titlesize': 11,
        'axes.titleweight': 'normal',
        'axes.labelsize': 10,
        'axes.labelweight': 'normal',
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        # 'figure.dpi': 600,
        # 'savefig.dpi': 600,
        # 'figure.figsize': (5, 5),  # Square, smaller figure
        # 'axes.linewidth': 0.5,
        # 'grid.linewidth': 0.5,
        # 'lines.linewidth': 1.2,
        # 'xtick.major.width': 0.5,
        # 'ytick.major.width': 0.5,
        # 'xtick.minor.width': 0.5,
        # 'ytick.minor.width': 0.5,
        # 'xtick.direction': 'out',
        # 'ytick.direction': 'out',
        # 'axes.spines.top': False,
        # 'axes.spines.right': False,
        'axes.grid': False,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
    })

    
def plot_confusion_matrix(true_labels, pred_labels, class_names=None, output_dir=None, task_name=None, mode='val', normalized=True):
    """
    Plot confusion matrix with probabilities or absolute values
    
    Parameters:
    -----------
    true_labels : array-like
        Ground truth labels
    pred_labels : array-like
        Predicted labels
    class_names : list, optional
        Names of the classes
    output_dir : str
        Directory to save the plots
    task_name : str
        Name of the task or model
    mode : str
        'val' or 'test' or 'train'
    normalized : bool
        Whether to normalize the confusion matrix
    """
    # Always use basic style for confusion matrices (not SciencePlots)
    set_basic_style()
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Create both probability and absolute figures
    for is_norm in [True, False]:
        fig, ax = plt.subplots(figsize=(4, 4))  # Square figure
        
        if is_norm:
            cm_plot = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            vmax = cm_plot.max()
            title_suffix = "Probability"
            file_suffix = "prob"
        else:
            cm_plot = cm
            fmt = 'd'
            vmax = None
            title_suffix = "Count"
            file_suffix = "count"
        
        sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues', 
                   square=True, cbar=True, ax=ax, vmax=vmax,
                   annot_kws={"size": 8}, cbar_kws={"shrink": 0.75})
        
        if class_names:
            ax.set_xticklabels(class_names)
            ax.set_yticklabels(class_names)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix ({title_suffix})')
        
        # Save the figure if output directory is provided
        if output_dir and task_name:
            os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
            filename = f'confusion_matrix_{mode}_{file_suffix}.pdf'
            filepath = os.path.join(output_dir, task_name, filename)
            fig.tight_layout()
            fig.savefig(filepath, format='pdf', bbox_inches='tight')
            print(f"Saved confusion matrix ({title_suffix}) to: {filepath}")
            plt.close(fig)


def plot_binary_roc(true_labels, pred_scores, output_dir=None, task_name=None, mode='test', plot_random=False):
    """
    绘制二分类ROC曲线及AUC填充(仅针对二分类).
    
    参数:
    -----
    true_labels : array-like of shape (N,)
        真实标签(0/1)
    pred_scores : array-like of shape (N,)
        模型预测的“正类”概率/置信度(例如 Sigmoid 或 Softmax 后针对正类的分数).
    output_dir  : str
        保存目录
    task_name   : str
        任务名称，用于区分不同任务/模型
    mode        : str
        'val' or 'test' or 'train'，用于文件命名
    """
    set_basic_style()

    fpr, tpr, thresholds = roc_curve(true_labels, pred_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.fill_between(fpr, tpr, alpha=0.3)
    if plot_random:
        ax.plot([0, 1], [0, 1], 'k--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right', handlelength=2.5)

    if output_dir and task_name:
        os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
        filename = f'roc_curve_{mode}.pdf'
        filepath = os.path.join(output_dir, task_name, filename)
        fig.tight_layout()
        fig.savefig(filepath, format='pdf', bbox_inches='tight')
        print(f"Saved binary ROC curve to: {filepath}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def plot_binary_pr(true_labels, pred_scores, output_dir=None, task_name=None, mode='test', plot_random=False):
    """
    绘制二分类PR曲线及面积填充(仅针对二分类).
    """
    set_basic_style()
    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    ap = average_precision_score(true_labels, pred_scores) # 手动微积分验证 pr_auc = np.trapz(precision, recall)
    
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(recall, precision, label=f'PR (AP = {ap:.3f}')
    ax.fill_between(recall, precision, alpha=0.3)

    if plot_random:
        pos_frac = np.mean(true_labels)
        ax.hlines(pos_frac, 0, 1, colors='k', linestyles='--', label=f'Random (pos = {pos_frac:.3f})')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR Curve')
    ax.legend(loc='lower right', handlelength=2.5)
    fig.tight_layout()

    if output_dir and task_name:
        os.makedirs(os.path.join(output_dir, task_name), exist_ok=True)
        filename = f'pr_curve_{mode}.pdf'
        filepath = os.path.join(output_dir, task_name, filename)
        fig.savefig(filepath, format='pdf', bbox_inches='tight')
        print(f"Saved binary PR curve to: {filepath}")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)
