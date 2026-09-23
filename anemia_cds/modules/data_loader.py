"""
data_loader.py — Load, validate, and activate real or synthetic CSV datasets.
"""
import os
import shutil
import pandas as pd
from rich.console import Console
from rich.table import Table
import config

console = Console()

REQUIRED_COLS = ['Hb', 'MCV', 'MCH', 'MCHC', 'RBC', 'RDW', 'Hematocrit', 'label']


class DataLoader:
    """Load and validate CBC CSV datasets, then activate for training."""

    def load(self, path: str) -> pd.DataFrame:
        """Load a CSV and validate structure + biologic plausibility."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)
        self._validate_columns(df)
        self._validate_labels(df)
        self._validate_ranges(df)

        console.print(f"[green]✓ Loaded {len(df)} rows from {path}[/green]")
        return df

    def _validate_columns(self, df: pd.DataFrame):
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}\nRequired: {REQUIRED_COLS}")

    def _validate_labels(self, df: pd.DataFrame):
        invalid = set(df['label'].unique()) - set(config.TARGET_CLASSES)
        if invalid:
            raise ValueError(
                f"Invalid label values: {invalid}\n"
                f"Allowed: {config.TARGET_CLASSES}"
            )

    def _validate_ranges(self, df: pd.DataFrame):
        """Flag (but don't fail) rows outside biologic plausibility ranges."""
        flags = []
        for col, (lo, hi) in config.BIOLOGIC_RANGES.items():
            if col in df.columns:
                out = df[(df[col] < lo) | (df[col] > hi)]
                if len(out) > 0:
                    flags.append(f"  {col}: {len(out)} rows outside [{lo}, {hi}]")
        if flags:
            console.print("[yellow]⚠ Biologic range warnings:[/yellow]")
            for f in flags:
                console.print(f"[yellow]{f}[/yellow]")

    def activate(self, path: str):
        """Copy dataset to data/active/cbc_active.csv for training."""
        os.makedirs(os.path.dirname(config.ACTIVE_CSV), exist_ok=True)
        shutil.copy(path, config.ACTIVE_CSV)
        console.print(f"[green]✓ Dataset activated: {path} → {config.ACTIVE_CSV}[/green]")

    def get_active(self) -> pd.DataFrame:
        """Load the currently active dataset."""
        return self.load(config.ACTIVE_CSV)

    def describe(self, df: pd.DataFrame):
        """Print a Rich table summary of the dataset."""
        table = Table(title="Dataset Summary", border_style="blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Rows", str(len(df)))
        table.add_row("Features", str(len(REQUIRED_COLS) - 1))
        table.add_row("Classes", str(df['label'].nunique()))

        for label, count in df['label'].value_counts().items():
            pct = count / len(df) * 100
            table.add_row(f"  ↳ {label}", f"{count} ({pct:.1f}%)")

        console.print(table)

    def list_real_datasets(self) -> list:
        """List all CSV files dropped into data/real/."""
        real_dir = os.path.join(config.DATA_DIR, 'real')
        os.makedirs(real_dir, exist_ok=True)
        return [
            os.path.join(real_dir, f)
            for f in os.listdir(real_dir)
            if f.endswith('.csv')
        ]
