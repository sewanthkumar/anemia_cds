"""
explainer.py — SHAP global and local explanations for any sklearn-compatible model.
Generates summary plots (global) and force/waterfall plots (per patient).
Runs entirely locally — no LLM calls.
"""
import os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file saving
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
import config

console = Console()

EDA_DIR = os.path.join(config.OUTPUT_DIR, 'eda')
XAI_DIR = os.path.join(config.OUTPUT_DIR, 'xai')


class Explainer:
    """
    SHAP-based explainability for trained models.
    - Global: feature importance summary plot
    - Local: per-patient SHAP waterfall / force plot
    """

    def __init__(self, model, preprocessor):
        """
        model: trained sklearn-compatible model
        preprocessor: fitted Preprocessor instance
        """
        self.model        = model
        self.preprocessor = preprocessor
        self.explainer    = None
        self.shap_values  = None

    def _get_explainer(self, X_background):
        """Create SHAP explainer appropriate to model type."""
        model_type = type(self.model).__name__

        if 'XGB' in model_type or 'LGBM' in model_type or 'Forest' in model_type:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Kernel explainer for linear/other models — use 50-sample background
            background = shap.sample(X_background, min(50, len(X_background)))
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )

    # ─────────────────────────── Global Explanation ───────────────────────────

    def global_explanation(self, X: np.ndarray, save: bool = True) -> str:
        """
        Compute SHAP values for full dataset and generate summary plot.
        Returns path to saved plot.
        """
        os.makedirs(XAI_DIR, exist_ok=True)
        self._get_explainer(X)

        console.print("[cyan]⚙ Computing SHAP values (global)...[/cyan]")
        self.shap_values = self.explainer.shap_values(X)

        # If 3D array (XGBoost multiclass: n_samples, n_features, n_classes), convert to list of 2D arrays
        if hasattr(self.shap_values, 'ndim') and self.shap_values.ndim == 3:
            self.shap_values = [self.shap_values[:, :, i] for i in range(self.shap_values.shape[2])]

        # For multi-class, shap_values is a list of arrays — use mean absolute
        if isinstance(self.shap_values, list):
            mean_abs = np.mean([np.abs(sv) for sv in self.shap_values], axis=0)
        else:
            mean_abs = np.abs(self.shap_values)

        # Feature importance table in terminal
        importance = mean_abs.mean(axis=0)
        self._print_feature_importance(importance)

        if save:
            out_path = os.path.join(XAI_DIR, 'shap_summary.png')
            plt.figure(figsize=(10, 6))
            if isinstance(self.shap_values, list):
                shap.summary_plot(
                    self.shap_values[0], X,
                    feature_names=config.FEATURES,
                    show=False, plot_size=(10, 6)
                )
            else:
                shap.summary_plot(
                    self.shap_values, X,
                    feature_names=config.FEATURES,
                    show=False, plot_size=(10, 6)
                )
            plt.title('SHAP Feature Importance — Global')
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            console.print(f"[green]✓ SHAP summary plot saved → {out_path}[/green]")
            return out_path

        return ""

    def _print_feature_importance(self, importance: np.ndarray):
        """Print SHAP feature importance as a Rich table."""
        table = Table(title="SHAP Feature Importance (Global)", border_style="magenta")
        table.add_column("Rank", style="dim")
        table.add_column("Feature", style="cyan")
        table.add_column("Mean |SHAP|", style="white")

        order = np.argsort(importance)[::-1]
        for rank, idx in enumerate(order, 1):
            bar = "█" * int(importance[idx] / (importance.max() + 1e-8) * 20)
            table.add_row(str(rank), config.FEATURES[idx], f"{importance[idx]:.4f}  {bar}")

        console.print(table)

    # ─────────────────────────── Local Explanation ────────────────────────────

    def local_explanation(self, X_patient: np.ndarray, patient_id: str = "patient") -> list:
        """
        Compute and display SHAP contribution for a single patient.
        Returns list of (feature, shap_value) tuples sorted by |value| desc.
        """
        if self.explainer is None:
            raise RuntimeError("Call global_explanation() first to initialize the SHAP explainer.")

        shap_vals = self.explainer.shap_values(X_patient)

        # Multi-class: pick the predicted class index
        pred_idx = self.model.predict(X_patient)[0]
        
        # Handle different SHAP output formats (3D array vs List vs 2D array)
        if hasattr(shap_vals, 'ndim') and shap_vals.ndim == 3:
            sv = shap_vals[0, :, pred_idx]
        elif isinstance(shap_vals, list):
            sv = shap_vals[pred_idx][0]
        else:
            sv = shap_vals[0]

        # Waterfall plot
        os.makedirs(XAI_DIR, exist_ok=True)
        out_path = os.path.join(XAI_DIR, f'shap_local_{patient_id}.png')

        plt.figure(figsize=(10, 5))
        colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in sv]
        plt.barh(config.FEATURES, sv, color=colors)
        plt.axvline(0, color='white', linewidth=0.8)
        plt.title(f'SHAP Local Explanation — {patient_id}')
        plt.xlabel('SHAP Value')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        contributions = sorted(
            zip(config.FEATURES, sv),
            key=lambda x: abs(x[1]), reverse=True
        )

        # Terminal display
        table = Table(title=f"Top SHAP Contributions — {patient_id}", border_style="red")
        table.add_column("Feature", style="cyan")
        table.add_column("SHAP Value", style="white")
        table.add_column("Direction", style="white")

        for feat, val in contributions[:5]:
            direction = "[red]↑ pushes toward prediction[/red]" if val > 0 else "[green]↓ pushes against[/green]"
            table.add_row(feat, f"{val:+.4f}", direction)

        console.print(table)
        console.print(f"[dim]Local SHAP plot saved → {out_path}[/dim]")

        return contributions

    def top_shap_features(self, X_patient: np.ndarray, n: int = 5) -> str:
        """Returns formatted string of top N SHAP features for clinical report."""
        contribs = self.local_explanation(X_patient)
        lines = [f"{feat} ({val:+.3f})" for feat, val in contribs[:n]]
        return ", ".join(lines)
