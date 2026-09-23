"""
preprocessor.py — Feature engineering, scaling, encoding, and SMOTE balancing.
Computes all derived hematologic indices and prepares data for ML training.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from rich.console import Console
import config

console = Console()


class Preprocessor:
    """
    Full preprocessing pipeline:
    1. Compute derived hematologic indices
    2. Handle missing values
    3. Encode labels
    4. Scale features
    5. Apply SMOTE for class balancing
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self._fitted = False

    # ─────────────────────────── Feature Engineering ─────────────────────────

    @staticmethod
    def compute_indices(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Mentzer, Shine-Lal, England-Fraser, RDW/MCV, Hb/RBC indices.
        Input df must have: Hb, MCV, MCH, MCHC, RBC, RDW, Hematocrit columns.
        Safe division to avoid divide-by-zero.
        """
        df = df.copy()

        rbc_safe = df['RBC'].replace(0, np.nan)
        df['Mentzer_Index']   = df['MCV'] / rbc_safe                      # >13 → IDA
        df['Shine_Lal']       = (df['MCV'] ** 2) * df['MCH'] / 100        # Red cell size index
        df['England_Fraser']  = df['MCV'] - rbc_safe - (5 * df['Hb'])      # Neg → Thal
        df['RDW_MCV_Ratio']   = df['RDW'] / df['MCV'].replace(0, np.nan)
        df['Hb_RBC_Ratio']    = df['Hb'] / rbc_safe

        # Fill any NaN from division with column medians
        for col in ['Mentzer_Index', 'Shine_Lal', 'England_Fraser', 'RDW_MCV_Ratio', 'Hb_RBC_Ratio']:
            df[col] = df[col].fillna(df[col].median())

        return df

    @staticmethod
    def compute_single_patient(cbc: dict) -> dict:
        """
        Compute derived indices for a single patient dict.
        Returns extended dict with all model features.
        """
        rbc = cbc.get('RBC', 1) or 1
        mcv = cbc.get('MCV', 1) or 1
        hb  = cbc.get('Hb', 0) or 0
        mch = cbc.get('MCH', 0) or 0

        return {
            **cbc,
            'Mentzer_Index':  mcv / rbc,
            'Shine_Lal':      (mcv ** 2) * mch / 100,
            'England_Fraser': mcv - rbc - (5 * hb),
            'RDW_MCV_Ratio':  cbc.get('RDW', 0) / mcv,
            'Hb_RBC_Ratio':   hb / rbc,
        }

    # ─────────────────────────── Full Training Pipeline ──────────────────────

    def fit_transform(self, df: pd.DataFrame):
        """
        Full fit pipeline for training:
        engineer → encode → split → scale → SMOTE → return X_train, X_test, y_train, y_test
        """
        df = self.compute_indices(df)

        # Drop rows with missing CBC values
        df = df.dropna(subset=config.FEATURES)

        X = df[config.FEATURES].values
        y = df[config.TARGET_COL].values

        # Encode labels
        y_enc = self.encoder.fit_transform(y)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y_enc
        )

        # Scale
        X_train = self.scaler.fit_transform(X_train)
        X_test  = self.scaler.transform(X_test)

        # SMOTE — balance training set only
        min_count = min(np.bincount(y_train))
        if min_count >= 2:
            k = min(5, min_count - 1)
            smote = SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            console.print(f"   SMOTE applied → training set: {len(X_train)} rows")
        else:
            console.print("[yellow]   SMOTE skipped — not enough samples per class[/yellow]")

        self._fitted = True
        return X_train, X_test, y_train, y_test

    def transform_single(self, cbc: dict) -> np.ndarray:
        """Transform a single patient dict → scaled feature vector."""
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit before transforming. Load a trained model first.")
        patient = self.compute_single_patient(cbc)
        row = np.array([[patient.get(f, 0) for f in config.FEATURES]])
        return self.scaler.transform(row)

    # ─────────────────────────── Persistence ─────────────────────────────────

    def save(self):
        """Save fitted scaler and encoder to models/."""
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        joblib.dump(self.scaler,  config.SCALER_PATH)
        joblib.dump(self.encoder, config.ENCODER_PATH)
        console.print(f"[green]✓ Scaler + encoder saved.[/green]")

    def load(self):
        """Load pre-fitted scaler and encoder."""
        self.scaler  = joblib.load(config.SCALER_PATH)
        self.encoder = joblib.load(config.ENCODER_PATH)
        self._fitted = True
