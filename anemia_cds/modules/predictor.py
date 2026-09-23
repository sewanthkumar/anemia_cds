"""
predictor.py — Load trained model and run inference on single or batch patients.
"""
import os
import numpy as np
import pandas as pd
import joblib
from rich.console import Console
from modules.preprocessor import Preprocessor
import config

console = Console()


class Predictor:
    """
    Load trained model + preprocessor and make predictions.
    Supports single patient dict and batch CSV prediction.
    """

    def __init__(self):
        self.model       = None
        self.preprocessor = Preprocessor()
        self._loaded     = False

    def load(self):
        """Load model, scaler, and encoder from models/."""
        for path, label in [
            (config.MODEL_PATH,   "model"),
            (config.SCALER_PATH,  "scaler"),
            (config.ENCODER_PATH, "label encoder"),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required {label} not found at {path}. "
                    "Please train a model first (option 6)."
                )

        self.model = joblib.load(config.MODEL_PATH)
        self.preprocessor.load()
        self._loaded = True
        console.print("[green]✓ Model loaded successfully.[/green]")

    def is_loaded(self) -> bool:
        return self._loaded

    # ─────────────────────────── Single Patient ───────────────────────────────

    def predict_one(self, cbc: dict) -> dict:
        """
        Predict anemia type for a single patient.
        cbc: dict with keys Hb, MCV, MCH, MCHC, RBC, RDW, Hematocrit
        Returns: {predicted_class, confidence, probabilities, derived_indices}
        """
        if not self._loaded:
            self.load()

        X = self.preprocessor.transform_single(cbc)
        pred_idx = self.model.predict(X)[0]

        # Probabilities
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X)[0]
        else:
            probs = np.zeros(len(config.TARGET_CLASSES))
            probs[pred_idx] = 1.0

        predicted_class = self.preprocessor.encoder.inverse_transform([pred_idx])[0]
        confidence      = float(probs[pred_idx]) * 100

        # Derived indices for clinical report
        from modules.preprocessor import Preprocessor
        indices = Preprocessor.compute_single_patient(cbc)

        prob_dict = {
            self.preprocessor.encoder.inverse_transform([i])[0]: round(float(p) * 100, 1)
            for i, p in enumerate(probs)
        }

        return {
            'predicted_class': predicted_class,
            'confidence':      round(confidence, 1),
            'probabilities':   prob_dict,
            'derived_indices': {
                'Mentzer_Index':  round(indices['Mentzer_Index'], 3),
                'Shine_Lal':      round(indices['Shine_Lal'], 2),
                'England_Fraser': round(indices['England_Fraser'], 3),
                'RDW_MCV_Ratio':  round(indices['RDW_MCV_Ratio'], 4),
                'Hb_RBC_Ratio':   round(indices['Hb_RBC_Ratio'], 3),
            }
        }

    # ─────────────────────────── Batch Prediction ────────────────────────────

    def predict_batch(self, csv_path: str) -> pd.DataFrame:
        """
        Predict all patients in a CSV file.
        CSV must have: Hb, MCV, MCH, MCHC, RBC, RDW, Hematocrit columns.
        Returns DataFrame with predictions appended.
        """
        if not self._loaded:
            self.load()

        df = pd.read_csv(csv_path)
        required = config.RAW_FEATURES
        missing  = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Batch CSV missing columns: {missing}")

        results = []
        for _, row in df.iterrows():
            cbc  = {k: row[k] for k in required if pd.notna(row.get(k))}
            pred = self.predict_one(cbc)
            results.append({
                'predicted_class': pred['predicted_class'],
                'confidence_pct':  pred['confidence'],
                **pred['probabilities'],
            })

        result_df = pd.DataFrame(results)
        output_df = pd.concat([df.reset_index(drop=True), result_df], axis=1)
        return output_df
