"""
data_generator.py — Synthetic CBC dataset generation.
Generates robust synthetic data programmatically to avoid LLM token limits and ensure 1500 full rows.
"""
import os
import numpy as np
import pandas as pd
from rich.console import Console
from modules.llm_client import LLMClient
import config

console = Console()

class DataGenerator:
    """Generates realistic synthetic CBC datasets using clinical distributions."""

    def __init__(self, client: LLMClient = None):
        self.client = client

    def generate(self, output_path: str = config.SYNTHETIC_CSV) -> pd.DataFrame:
        """
        Generates 1500 synthetic patient rows programmatically and saves to CSV.
        Classes: IDA 60%, Thalassemia_Trait 30%, Anemia_of_Chronic_Disease 10%.
        """
        console.print("[bold cyan]⚡ Generating synthetic CBC dataset (Simulation)...[/bold cyan]")
        
        n_patients = 1500
        n_ida = int(n_patients * 0.6)
        n_thal = int(n_patients * 0.3)
        n_acd = n_patients - n_ida - n_thal
        
        # Clinical rules based distributions
        # IDA: RDW high >15, RBC low-normal, MCV 60-79
        ida = pd.DataFrame({
            'patient_id': [f"P_{i}" for i in range(n_ida)],
            'age': np.random.normal(45, 15, n_ida).clip(18, 90).astype(int),
            'gender': np.random.choice(['M', 'F'], n_ida),
            'Hb': np.random.normal(9.5, 1.5, n_ida).clip(5, 11.5),
            'MCV': np.random.normal(70, 5, n_ida).clip(60, 79),
            'MCH': np.random.normal(23, 2, n_ida).clip(15, 26),
            'MCHC': np.random.normal(30, 1.5, n_ida).clip(26, 31.5),
            'RBC': np.random.normal(3.8, 0.4, n_ida).clip(2.5, 4.5),
            'RDW': np.random.normal(16.5, 1.5, n_ida).clip(15.1, 22.0),
            'Hematocrit': np.random.normal(30, 3, n_ida).clip(20, 35),
            'label': 'IDA'
        })

        # Thalassemia Trait: RDW normal <15, RBC high >5.0, MCV very low 60-75
        thal = pd.DataFrame({
            'patient_id': [f"P_{i + n_ida}" for i in range(n_thal)],
            'age': np.random.normal(35, 15, n_thal).clip(18, 90).astype(int),
            'gender': np.random.choice(['M', 'F'], n_thal),
            'Hb': np.random.normal(11.5, 1.0, n_thal).clip(10, 13.5),
            'MCV': np.random.normal(68, 3, n_thal).clip(60, 75),
            'MCH': np.random.normal(22, 1.5, n_thal).clip(18, 25),
            'MCHC': np.random.normal(32, 1.0, n_thal).clip(31, 35),
            'RBC': np.random.normal(5.8, 0.4, n_thal).clip(5.1, 7.0),
            'RDW': np.random.normal(13.5, 0.8, n_thal).clip(11.0, 14.5),
            'Hematocrit': np.random.normal(34, 2, n_thal).clip(30, 40),
            'label': 'Thalassemia_Trait'
        })

        # Anemia of Chronic Disease: MCV 72-85, RBC low-normal, RDW normal
        acd = pd.DataFrame({
            'patient_id': [f"P_{i + n_ida + n_thal}" for i in range(n_acd)],
            'age': np.random.normal(65, 10, n_acd).clip(30, 90).astype(int),
            'gender': np.random.choice(['M', 'F'], n_acd),
            'Hb': np.random.normal(10.5, 1.0, n_acd).clip(9, 12.5),
            'MCV': np.random.normal(78, 3, n_acd).clip(72, 85),
            'MCH': np.random.normal(26, 1.5, n_acd).clip(24, 29),
            'MCHC': np.random.normal(32, 1.0, n_acd).clip(31, 34),
            'RBC': np.random.normal(3.5, 0.3, n_acd).clip(2.5, 4.0),
            'RDW': np.random.normal(13.8, 0.5, n_acd).clip(12.0, 14.5),
            'Hematocrit': np.random.normal(32, 2, n_acd).clip(28, 36),
            'label': 'Anemia_of_Chronic_Disease'
        })
        
        # Add 15% overlap/noise
        df = pd.concat([ida, thal, acd], ignore_index=True)
        # Apply slight random noise
        for col in ['Hb', 'MCV', 'MCH', 'MCHC', 'RBC', 'RDW', 'Hematocrit']:
            noise = np.random.normal(0, df[col].std() * 0.05, len(df))
            df[col] += noise

        # Round to 2 decimal places
        numeric_cols = ['Hb', 'MCV', 'MCH', 'MCHC', 'RBC', 'RDW', 'Hematocrit']
        df[numeric_cols] = df[numeric_cols].round(2)
        
        # Shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        console.print(f"[green]✓ Dataset generated: {len(df)} rows → saved to {output_path}[/green]")
        class_dist = df['label'].value_counts().to_dict()
        console.print(f"  Class distribution: {class_dist}")

        return df
