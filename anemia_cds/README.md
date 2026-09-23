# HematoAI v3.0 — Explainable AI for Microcytic Anemia CDS

> A Rich terminal-based clinical decision support system for classifying microcytic anemia (IDA, Thalassemia Trait, Anemia of Chronic Disease) using local ML + cloud/local LLM explanations.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
cd anemia_cds
pip install -r requirements.txt
```

### 2. Install Tesseract OCR (for image parsing)
- **Linux/Mac:** `sudo apt install tesseract-ocr`
- **Windows:** Download from [GitHub UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

### 3. Configure API key
Edit `.env` and add your free Groq key:
```
GROQ_API_KEY=your_key_here
```
Get a free key at: https://console.groq.com

### 4. Run the app
```bash
python main.py
```

### 5. First run — generate data and train
- Press `6` → Train / Retrain Model
- This will auto-generate 1500 synthetic CBC patients via Groq, then train 4 ML models with Optuna tuning (~3 min)

### 6. Test predictions
- `3` — Manual CBC entry
- `1` — Upload a PDF lab report
- `2` — Upload a photo of a CBC report

---

## 🏗 Architecture

```
anemia_cds/
├── main.py                    # Rich terminal app — main menu loop
├── config.py                  # All constants
├── .env                       # GROQ_API_KEY (never commit)
├── requirements.txt
├── modules/
│   ├── llm_client.py          # ★ Single unified LLM interface (Groq + Ollama)
│   ├── data_generator.py      # Synthetic CBC generation via LLM
│   ├── data_loader.py         # Load/validate real CSV datasets
│   ├── report_parser.py       # PDF + image CBC extraction
│   ├── preprocessor.py        # Feature engineering + SMOTE
│   ├── trainer.py             # Local ML training + Optuna
│   ├── predictor.py           # Inference (single + batch)
│   ├── explainer.py           # SHAP global + local
│   ├── clinical_engine.py     # LLM clinical reports
│   ├── ui_components.py       # Rich panels/tables/menus
│   └── settings_manager.py   # Runtime settings (Groq ↔ Ollama)
├── data/
│   ├── synthetic/             # Auto-generated training data
│   ├── real/                  # Drop real CSVs here
│   └── active/                # Currently active dataset
├── models/                    # Saved model + metadata
├── outputs/
│   ├── eda/                   # EDA plots
│   └── xai/                   # SHAP plots
└── uploads/                   # Temp folder for PDF/image inputs
```

---

## 🔀 Groq → Ollama Switch

No code changes needed. From the app:
**Main Menu → S (Settings) → LLM Backend → ollama**

Then run:
```bash
ollama serve          # in a separate terminal
ollama pull llama3.2  # 2 GB — recommended text model
ollama pull llava     # 4.7 GB — required for image parsing
```

---

## 🤖 Models Trained Locally

| Model | Notes |
|-------|-------|
| XGBoost (Optuna tuned) | Primary — best for tabular CBC data |
| Random Forest | Robust baseline |
| LightGBM | Fast alternative |
| Logistic Regression | Interpretable sanity check |

Training time: ~3 min for 1500 rows × 50 Optuna trials on a 16 GB CPU machine. No GPU needed.

---

## 📊 CBC Features Used

| Feature | Description |
|---------|-------------|
| Hb | Hemoglobin |
| MCV | Mean Corpuscular Volume |
| MCH | Mean Corpuscular Hemoglobin |
| MCHC | Mean Corpuscular Hemoglobin Concentration |
| RBC | Red Blood Cell count |
| RDW | Red Cell Distribution Width |
| Hematocrit | Packed Cell Volume |
| Mentzer Index | MCV/RBC — >13 suggests IDA |
| Shine-Lal | MCV² × MCH / 100 |
| England-Fraser | MCV − RBC − 5×Hb |
| RDW/MCV Ratio | Discriminates IDA from Thal |
| Hb/RBC Ratio | Similar to MCH — cross-check |

---

## 📦 Real Dataset Format

Place CSV files in `data/real/`. Required columns:

```
Hb, MCV, MCH, MCHC, RBC, RDW, Hematocrit, label
```

Labels must be: `IDA`, `Thalassemia_Trait`, `Anemia_of_Chronic_Disease`
