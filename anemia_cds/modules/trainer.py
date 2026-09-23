"""
trainer.py — Local ML training with XGBoost, Random Forest, LightGBM,
Logistic Regression + Optuna hyperparameter tuning (Bayesian TPE, 5-fold CV).
Runs 100% locally — no LLM calls here.
"""
import os
import json
import warnings
import numpy as np
import joblib
import optuna
import optuna.logging
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix, accuracy_score
)
import xgboost as xgb
import lightgbm as lgb
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import config

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

console = Console()


class Trainer:
    """
    Local ML trainer — trains 4 model families, tunes best with Optuna,
    saves best model + metadata to models/.
    """

    def __init__(self):
        self.best_model = None
        self.best_name  = None
        self.best_f1    = 0.0
        self.best_accuracy = 0.0
        self.results    = {}

    # ─────────────────────────── Main Training Entry ─────────────────────────

    def train(self, X_train, X_test, y_train, y_test) -> dict:
        """
        Full training pipeline:
        1. Quick baseline: RF, LightGBM, LogReg
        2. Tune XGBoost with Optuna
        3. Select best by F1-macro
        4. Final evaluation on test set
        Returns results dict with metrics for all models.
        """
        console.print("\n[bold cyan]🤖 Training ML Models (locally)...[/bold cyan]")

        skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        n_classes = len(np.unique(y_train))

        # ── Baseline models ───────────────────────────────────────────────────
        baselines = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=config.RANDOM_STATE, n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05,
                random_state=config.RANDOM_STATE, verbose=-1
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, random_state=config.RANDOM_STATE
            ),
        }

        for name, model in baselines.items():
            with Progress(SpinnerColumn(), TextColumn(f"  Training {name}..."), transient=True) as p:
                p.add_task("train")
                scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)
                f1_cv = scores.mean()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1_test = f1_score(y_test, y_pred, average='macro')
            accuracy_test = accuracy_score(y_test, y_pred)

            self.results[name] = {'cv_f1': round(f1_cv, 4), 'test_f1': round(f1_test, 4), 'test_accuracy': round(accuracy_test, 4)}
            console.print(f"  {name}: CV F1={f1_cv:.4f} | Test F1={f1_test:.4f} | Acc={accuracy_test:.4f}")

            if f1_test > self.best_f1:
                self.best_f1    = f1_test
                self.best_accuracy = accuracy_test
                self.best_model = model
                self.best_name  = name

        # ── XGBoost with Optuna ───────────────────────────────────────────────
        console.print(f"\n  [bold yellow]🔬 Tuning XGBoost with Optuna ({config.OPTUNA_TRIALS} trials)...[/bold yellow]")
        best_xgb, xgb_cv_f1 = self._tune_xgboost(X_train, y_train, skf, n_classes)

        best_xgb.fit(X_train, y_train)
        y_pred_xgb = best_xgb.predict(X_test)
        xgb_test_f1 = f1_score(y_test, y_pred_xgb, average='macro')
        xgb_test_accuracy = accuracy_score(y_test, y_pred_xgb)

        self.results['XGBoost (Optuna)'] = {
            'cv_f1': round(xgb_cv_f1, 4),
            'test_f1': round(xgb_test_f1, 4),
            'test_accuracy': round(xgb_test_accuracy, 4)
        }
        console.print(f"  XGBoost (Optuna): CV F1={xgb_cv_f1:.4f} | Test F1={xgb_test_f1:.4f} | Acc={xgb_test_accuracy:.4f}")

        if xgb_test_f1 > self.best_f1:
            self.best_f1    = xgb_test_f1
            self.best_accuracy = xgb_test_accuracy
            self.best_model = best_xgb
            self.best_name  = 'XGBoost (Optuna)'

        # ── Final report ──────────────────────────────────────────────────────
        y_pred_best = self.best_model.predict(X_test)
        self._print_results_table()
        self._print_classification_report(y_test, y_pred_best)

        return self.results

    def _tune_xgboost(self, X_train, y_train, skf, n_classes):
        """Optuna Bayesian search for best XGBoost hyperparameters."""

        def objective(trial):
            params = {
                'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
                'max_depth':         trial.suggest_int('max_depth', 3, 10),
                'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
                'gamma':             trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda':        trial.suggest_float('reg_lambda', 0.0, 2.0),
                'objective':         'multi:softmax' if n_classes > 2 else 'binary:logistic',
                'num_class':         n_classes if n_classes > 2 else None,
                'random_state':      config.RANDOM_STATE,
                'n_jobs':            -1,
                'eval_metric':       'mlogloss',
                'use_label_encoder': False,
            }
            if n_classes <= 2:
                del params['num_class']

            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=1)
            return scores.mean()

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

        with Progress(SpinnerColumn(), TextColumn("  Optuna trials: {task.fields[progress]}"), console=console) as progress:
            task = progress.add_task("optuna", progress="0/" + str(config.OPTUNA_TRIALS))

            def callback(study, trial):
                progress.update(task, progress=f"{trial.number + 1}/{config.OPTUNA_TRIALS}")

            study.optimize(objective, n_trials=config.OPTUNA_TRIALS, callbacks=[callback], show_progress_bar=False)

        best_params = study.best_params
        best_params.update({
            'objective':   'multi:softmax' if n_classes > 2 else 'binary:logistic',
            'random_state': config.RANDOM_STATE,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
        })
        if n_classes > 2:
            best_params['num_class'] = n_classes

        return xgb.XGBClassifier(**best_params), study.best_value

    # ─────────────────────────── Display ─────────────────────────────────────

    def _print_results_table(self):
        table = Table(title="Model Comparison", border_style="green")
        table.add_column("Model", style="cyan")
        table.add_column("CV F1 (macro)", style="white")
        table.add_column("Test F1 (macro)", style="white")
        table.add_column("Status", style="white")

        for name, r in self.results.items():
            is_best = name == self.best_name
            status  = "[bold green]★ BEST[/bold green]" if is_best else ""
            table.add_row(name, f"{r['cv_f1']:.4f}", f"{r['test_f1']:.4f}", status)

        console.print(table)

    def _print_classification_report(self, y_test, y_pred):
        console.print("\n[bold]Classification Report (Best Model):[/bold]")
        console.print(classification_report(y_test, y_pred))

    # ─────────────────────────── Persistence ─────────────────────────────────

    def save(self, dataset_info: dict):
        """Save best model + metadata to models/."""
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(self.best_model, config.MODEL_PATH)

        metadata = {
            'model_name':   self.best_name,
            'f1_macro':     round(self.best_f1, 4),
            'accuracy':     round(self.best_accuracy, 4),
            'trained_at':   datetime.now().strftime('%Y-%m-%d %H:%M'),
            'dataset':      dataset_info,
            'features':     config.FEATURES,
            'classes':      config.TARGET_CLASSES,
        }
        with open(config.METADATA_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)

        console.print(f"[green]✓ Best model saved: {self.best_name} (F1={self.best_f1:.4f})[/green]")

    def load_metadata(self) -> dict:
        if not os.path.exists(config.METADATA_PATH):
            return {}
        with open(config.METADATA_PATH) as f:
            return json.load(f)
