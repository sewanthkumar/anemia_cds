"""
main.py — HematoAI v3.0 Entry Point
Rich terminal app — main menu loop for Microcytic Anemia CDS.
Run with: python main.py
"""
import os
import sys
import json

# ── Ensure we run from the anemia_cds/ directory ─────────────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel

import config
from modules.llm_client       import LLMClient
from modules.ui_components    import print_header, print_menu, prompt_cbc_manual, print_model_performance, run_eda_plots, print_batch_results, run_with_jarvis_spinner
from modules.settings_manager import SettingsManager
from modules.trainer          import Trainer
from modules.preprocessor     import Preprocessor
from modules.predictor        import Predictor
from modules.data_loader      import DataLoader
from modules.data_generator   import DataGenerator
from modules.report_parser    import ReportParser
from modules.clinical_engine  import ClinicalEngine
from modules.explainer        import Explainer

console = Console()


def _ensure_dirs():
    """Create required directories if they don't exist."""
    for d in [config.MODEL_DIR, config.DATA_DIR, config.OUTPUT_DIR, config.UPLOAD_DIR,
              'data/synthetic', 'data/real', 'data/active', 'outputs/eda', 'outputs/xai']:
        os.makedirs(d, exist_ok=True)


def _init_llm() -> LLMClient:
    """Initialize LLM client from saved settings or config default."""
    settings_path = config.SETTINGS_PATH
    backend = config.LLM_BACKEND

    if os.path.exists(settings_path):
        with open(settings_path) as f:
            saved = json.load(f)
        backend = saved.get('backend', backend)

    try:
        client = LLMClient(backend=backend)
        # Restore saved model choice
        if os.path.exists(settings_path):
            if backend == 'groq' and saved.get('groq_model'):
                client.GROQ_TEXT_MODEL = saved['groq_model']
            elif backend == 'ollama' and saved.get('ollama_model'):
                client.OLLAMA_TEXT_MODEL = saved['ollama_model']
        return client
    except (ValueError, ConnectionError) as e:
        console.print(f"[yellow]⚠ LLM backend '{backend}' unavailable: {e}[/yellow]")
        console.print("[yellow]  Falling back to Groq. Add GROQ_API_KEY to .env[/yellow]")
        # Return a stub that will error gracefully on first use
        return None


# ─────────────────────────── Menu Handlers ───────────────────────────────────

def handle_predict_pdf(predictor, parser, clinical, explainer_obj):
    path = Prompt.ask("[cyan]Enter path to PDF lab report[/cyan]")
    path = path.strip().strip('"')
    try:
        cbc = parser.parse_pdf(path)
        _run_prediction(cbc, predictor, clinical, explainer_obj)
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")


def handle_predict_image(predictor, parser, clinical, explainer_obj):
    path = Prompt.ask("[cyan]Enter path to image file (jpg/png)[/cyan]")
    path = path.strip().strip('"')
    try:
        cbc = parser.parse_image(path)
        _run_prediction(cbc, predictor, clinical, explainer_obj)
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")


def handle_predict_manual(predictor, clinical, explainer_obj):
    try:
        cbc = prompt_cbc_manual()
        # Fill missing values with midpoints
        for feat, (lo, hi) in config.BIOLOGIC_RANGES.items():
            if cbc.get(feat) is None:
                cbc[feat] = (lo + hi) / 2
        _run_prediction(cbc, predictor, clinical, explainer_obj)
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")


def _run_prediction(cbc: dict, predictor: Predictor, clinical: ClinicalEngine, explainer_obj):
    """Run full prediction + SHAP + clinical report for a single patient."""
    if not predictor.is_loaded():
        try:
            predictor.load()
        except FileNotFoundError as e:
            console.print(f"[red]✗ {e}[/red]")
            return

    result = predictor.predict_one(cbc)
    predicted_class = result['predicted_class']
    confidence      = result['confidence']
    probabilities   = result['probabilities']
    derived_indices = result['derived_indices']

    # SHAP local explanation
    shap_features = "N/A"
    if explainer_obj and explainer_obj.explainer:
        try:
            X = predictor.preprocessor.transform_single(cbc)
            shap_features = explainer_obj.top_shap_features(X)
        except Exception:
            pass

    # Clinical report
    explanation = "No LLM explanation available."
    action      = ""
    if clinical:
        try:
            def fetch_llm():
                exp = clinical.clinical_explanation(
                    cbc, predicted_class, confidence, derived_indices, shap_features
                )
                act = clinical.action_justification(predicted_class, confidence, cbc)
                return exp, act
            
            explanation, action = run_with_jarvis_spinner("QUERYING HEMATOLOGY LLM...", fetch_llm)
        except Exception as e:
            console.print(f"[yellow]⚠ LLM unavailable: {e}[/yellow]")

    clinical.display_report(
        predicted_class, confidence, probabilities,
        cbc, derived_indices, explanation, action
    ) if clinical else None

    if clinical:
        export = Prompt.ask("\n[bold cyan]Export patient report as Markdown? [y/n][/bold cyan]", default="n").strip().lower()
        if export == 'y':
            report_path = clinical.export_markdown_report(
                predicted_class, confidence, probabilities,
                cbc, derived_indices, explanation, action
            )
            console.print(f"[green]✓ Report exported to {report_path}[/green]")

def handle_batch_csv(predictor):
    path = Prompt.ask("[cyan]Enter path to batch CSV file[/cyan]")
    path = path.strip().strip('"')
    try:
        df = predictor.predict_batch(path)
        print_batch_results(df)

        save = Prompt.ask("Save results to CSV? [y/n]", default="y")
        if save.lower() == 'y':
            out = path.replace('.csv', '_predictions.csv')
            df.to_csv(out, index=False)
            console.print(f"[green]✓ Results saved → {out}[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")


def handle_eda(loader, clinical, client):
    """Run EDA on active dataset — plots + LLM narrative."""
    try:
        df = loader.get_active()
    except FileNotFoundError:
        console.print("[yellow]No active dataset. Train first (option 6) or load real data (option 7).[/yellow]")
        return

    loader.describe(df)

    # Compute per-class stats for LLM narrative
    stats = df.groupby('label')[config.RAW_FEATURES].mean().round(2).to_dict()

    narrative = None
    if client:
        try:
            narrative = run_with_jarvis_spinner("ANALYZING EDA WITH LLM...", clinical.eda_narrative, stats)
        except Exception as e:
            console.print(f"[yellow]⚠ LLM narrative unavailable: {e}[/yellow]")

    run_eda_plots(df, narrative)


def handle_train(loader, client):
    """Generate synthetic data (if needed) + train model."""
    # Check for active dataset
    if not os.path.exists(config.ACTIVE_CSV):
        console.print("[yellow]No active dataset found. Generating synthetic data first...[/yellow]")
        if client is None:
            console.print("[red]✗ LLM not initialized. Cannot generate synthetic data.[/red]")
            return
        gen = DataGenerator(client)
        try:
            df = run_with_jarvis_spinner("GENERATING SYNTHETIC DATA...", gen.generate)
        except Exception as e:
            console.print(f"[red]✗ Data generation failed: {e}[/red]")
            return
        loader.activate(config.SYNTHETIC_CSV)
    else:
        console.print(f"[green]Using active dataset: {config.ACTIVE_CSV}[/green]")

    try:
        df = loader.get_active()
    except Exception as e:
        console.print(f"[red]✗ Could not load dataset: {e}[/red]")
        return

    loader.describe(df)

    # Preprocess
    preprocessor = Preprocessor()
    try:
        X_train, X_test, y_train, y_test = preprocessor.fit_transform(df)
    except Exception as e:
        console.print(f"[red]✗ Preprocessing failed: {e}[/red]")
        return

    # Train
    trainer = Trainer()
    try:
        run_with_jarvis_spinner("TRAINING ML MODEL...", trainer.train, X_train, X_test, y_train, y_test)
    except Exception as e:
        console.print(f"[red]✗ Training failed: {e}[/red]")
        return

    # Save
    dataset_info = {'source': 'synthetic', 'rows': len(df)}
    if not os.path.exists(config.SYNTHETIC_CSV) or config.ACTIVE_CSV != config.SYNTHETIC_CSV:
        dataset_info['source'] = 'real'

    trainer.save(dataset_info)
    preprocessor.save()
    console.print("\n[bold green]✓ Training complete! Model saved to models/[/bold green]")

    # Global SHAP
    run_shap = Prompt.ask("Run global SHAP explanation now? [y/n]", default="y")
    if run_shap.lower() == 'y':
        try:
            exp = Explainer(trainer.best_model, preprocessor)
            exp.global_explanation(X_test)
        except Exception as e:
            console.print(f"[yellow]⚠ SHAP failed: {e}[/yellow]")


def handle_load_dataset(loader):
    """Let user choose a real CSV from data/real/ and activate it."""
    files = loader.list_real_datasets()
    if not files:
        console.print(f"[yellow]No CSV files found in data/real/[/yellow]")
        console.print(f"  Drop your CSV files into [cyan]data/real/[/cyan] and try again.")
        console.print(f"  Required columns: Hb, MCV, MCH, MCHC, RBC, RDW, Hematocrit, label")
        return

    console.print("\n  Available datasets:")
    for i, f in enumerate(files, 1):
        console.print(f"    [cyan]{i}.[/cyan] {os.path.basename(f)}")

    idx = Prompt.ask("Choose dataset", choices=[str(i) for i in range(1, len(files)+1)])
    selected = files[int(idx) - 1]

    try:
        df = loader.load(selected)
        loader.describe(df)
        loader.activate(selected)
        console.print("[green]✓ Dataset loaded and activated.[/green]")
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")


def handle_shap(predictor):
    """Run global SHAP explanation on the active dataset."""
    if not predictor.is_loaded():
        try:
            predictor.load()
        except Exception as e:
            console.print(f"[red]✗ {e}[/red]")
            return

    try:
        df_loader = DataLoader()
        df = df_loader.get_active()
        df = Preprocessor.compute_indices(df)
        df = df.dropna(subset=config.FEATURES)
        X = predictor.preprocessor.scaler.transform(df[config.FEATURES].values)
    except Exception as e:
        console.print(f"[red]✗ Could not load dataset for SHAP: {e}[/red]")
        return

    explainer = Explainer(predictor.model, predictor.preprocessor)
    try:
        explainer.global_explanation(X)
    except Exception as e:
        console.print(f"[red]✗ SHAP failed: {e}[/red]")


# ─────────────────────────── Main Loop ───────────────────────────────────────

def main():
    _ensure_dirs()
    console.clear()
    
    from modules.ui_components import play_boot_animation
    play_boot_animation()

    # Initialize LLM client
    with console.status("[bold red]Connecting to AI Core...[/bold red]"):
        client = _init_llm()

    # Initialize shared objects
    loader   = DataLoader()
    clinical = ClinicalEngine(client) if client else None
    predictor = Predictor()
    parser   = ReportParser(client) if client else None
    settings = SettingsManager(client) if client else None
    explainer_obj = None  # Will be set after training

    trainer_meta = Trainer().load_metadata()
    llm_status   = client.get_status() if client else {'backend': 'Not initialized', 'text_model': 'N/A', 'vision_model': 'N/A'}
    print_header(llm_status, trainer_meta)
    print_menu()

    while True:
        trainer_meta = Trainer().load_metadata()
        llm_status   = client.get_status() if client else {'backend': 'Not initialized', 'text_model': 'N/A', 'vision_model': 'N/A'}

        choice = Prompt.ask("\n[bold cyan]Select option[/bold cyan] (Type 'M' to show menu)", default="3").strip().upper()

        if choice == 'M':
            print_header(llm_status, trainer_meta)
            print_menu()
            continue

        if choice == '1':
            if not client:
                console.print("[red]LLM not initialized. Add GROQ_API_KEY to .env[/red]")
            else:
                handle_predict_pdf(predictor, parser, clinical, explainer_obj)

        elif choice == '2':
            if not client:
                console.print("[red]LLM not initialized.[/red]")
            else:
                handle_predict_image(predictor, parser, clinical, explainer_obj)

        elif choice == '3':
            handle_predict_manual(predictor, clinical, explainer_obj)

        elif choice == '4':
            handle_batch_csv(predictor)

        elif choice == '5':
            handle_eda(loader, clinical, client)

        elif choice == '6':
            handle_train(loader, client)

        elif choice == '7':
            handle_load_dataset(loader)

        elif choice == '8':
            print_model_performance(trainer_meta)

        elif choice == '9':
            handle_shap(predictor)

        elif choice == 'S':
            if settings:
                settings.show_settings_menu()
                # Re-sync client after settings change
                client   = settings.client
                clinical = ClinicalEngine(client)
                parser   = ReportParser(client)
            else:
                console.print("[yellow]LLM not initialized — cannot open settings.[/yellow]")

        elif choice == 'Q':
            console.print("\n[bold red]Goodbye. Stay healthy! 🩺[/bold red]\n")
            sys.exit(0)

        else:
            console.print(f"[yellow]Unknown option '{choice}'. Please select from the menu.[/yellow]")


if __name__ == '__main__':
    main()
