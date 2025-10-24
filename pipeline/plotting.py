"""
Plotting utilities: confusion matrix, ROC curves, and feature importance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], out_path: Path) -> None:
    """
    Plot confusion matrix using sklearn ConfusionMatrixDisplay.
    
    Args:
        cm: Confusion matrix as numpy array
        labels: List of class labels
        out_path: Path to save the plot
    """
    try:
        # Use sklearn ConfusionMatrixDisplay for better visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create display object
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        display.plot(ax=ax, cmap='viridis', colorbar=True)
        
        # Customize the plot
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"Warning: Could not create confusion matrix plot: {e}")
        # Fallback to simple plot
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap='viridis')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), 
               xticklabels=labels, yticklabels=labels, 
               ylabel='True label', xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(int(cm[i, j])), ha="center", va="center", 
                       color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, class_names: Optional[List[str]] = None) -> None:
    """
    Plot ROC curves using sklearn RocCurveDisplay.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        out_path: Path to save the plot
        class_names: Optional list of class names
    """
    try:
        n_classes = y_prob.shape[1]
        
        # Handle binary classification
        if n_classes == 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Use RocCurveDisplay for binary classification
            display = RocCurveDisplay.from_predictions(
                y_true, y_prob[:, 1], 
                name=f"ROC Curve (AUC = {auc(roc_curve(y_true, y_prob[:, 1])[0], roc_curve(y_true, y_prob[:, 1])[1]):.3f})",
                ax=ax
            )
            
        else:
            # Multi-class classification
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Binarize the labels for multi-class ROC
            label_binarizer = LabelBinarizer().fit(y_true)
            y_onehot = label_binarizer.transform(y_true)
            
            # Plot ROC curve for each class
            for i in range(n_classes):
                class_name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
                
                if y_onehot[:, i].sum() > 0:  # Only plot if class exists in test set
                    fpr, tpr, _ = roc_curve(y_onehot[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    
                    ax.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.3f})")
            
            # Plot diagonal line
            ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        # Customize the plot
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"Warning: Could not create ROC curves plot: {e}")
        # Fallback to simple plot
        fig, ax = plt.subplots(figsize=(6, 4))
        n_classes = y_prob.shape[1]
        for i in range(n_classes):
            y_bin = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
            ax.plot(fpr, tpr, label=f"Class {class_names[i] if class_names else i} (AUC={auc(fpr,tpr):.2f})")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_feature_importance(model: Any, feature_names: List[str], out_path: Path, max_features: int = 20) -> None:
    """
    Plot feature importance with improved visualization.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        out_path: Path to save the plot
        max_features: Maximum number of features to display
    """
    try:
        if not hasattr(model, "feature_importances_"):
            print("Warning: Model does not have feature_importances_ attribute")
            return
        
        importances = model.feature_importances_
        
        # Sort features by importance (descending)
        idx = np.argsort(importances)[::-1]
        
        # Limit to max_features for readability
        n_features = min(max_features, len(importances))
        top_indices = idx[:n_features]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create horizontal bar plot for better readability
        y_pos = np.arange(n_features)
        bars = ax.barh(y_pos, importances[top_indices], color='skyblue', alpha=0.7)
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in top_indices], fontsize=10)
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title(f'Top {n_features} Feature Importances', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances[top_indices])):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{importance:.3f}', ha='left', va='center', fontsize=9)
        
        # Invert y-axis to show most important features at the top
        ax.invert_yaxis()
        
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    except Exception as e:
        print(f"Warning: Could not create feature importance plot: {e}")
        # Fallback to simple plot
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            idx = np.argsort(importances)[::-1]
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(range(len(importances)), importances[idx])
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in idx], rotation=90)
            ax.set_ylabel('Importance')
            fig.tight_layout()
            fig.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)


def generate_all_plots(model: Any, results: Dict[str, Any], X_test: Any, y_test: Any, 
                       feature_names: List[str], exp_dir: Path, 
                       metrics: List[str], cv_mode: str = "groupkfold") -> None:
    """
    Generate all plots for the experiment.
    
    Args:
        model: Trained model
        results: Evaluation results dictionary
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        exp_dir: Experiment directory to save plots
        metrics: List of requested metrics
        cv_mode: Cross-validation mode ("groupkfold" or "loso")
    """
    print("\n📊 Generating plots...")
    
    # 1. Confusion Matrix
    if "confusion_matrix" in results:
        try:
            cm = np.array(results["confusion_matrix"], dtype=int)
            labels = sorted([str(c) for c in sorted(np.unique(y_test))])
            plot_confusion_matrix(cm, labels, exp_dir / "confusion_matrix.png")
            print("  ✅ Confusion matrix saved")
        except Exception as e:
            print(f"  ⚠️  Could not create confusion matrix: {e}")
    
    # 2. ROC Curves (only if roc_auc is requested and model supports predict_proba)
    if "roc_auc" in metrics and hasattr(model, "predict_proba"):
        try:
            if cv_mode.lower() == "loso":
                # For LOSO, we might not have separate test data
                print("  ⚠️  ROC curves skipped for LOSO mode (no separate test set)")
            else:
                y_prob = model.predict_proba(X_test)
                labels = sorted([str(c) for c in sorted(np.unique(y_test))])
                plot_roc_curves(y_test.to_numpy(), y_prob, exp_dir / "roc_curves.png", labels)
                print("  ✅ ROC curves saved")
        except Exception as e:
            print(f"  ⚠️  Could not create ROC curves: {e}")
    else:
        if "roc_auc" not in metrics:
            print("  ⚠️  ROC curves skipped (roc_auc not in metrics)")
        else:
            print("  ⚠️  ROC curves skipped (model does not support predict_proba)")
    
    # 3. Feature Importance
    try:
        plot_feature_importance(model, feature_names, exp_dir / "feature_importance.png")
        print("  ✅ Feature importance saved")
    except Exception as e:
        print(f"  ⚠️  Could not create feature importance plot: {e}")
    
    print("📊 Plot generation completed!")


