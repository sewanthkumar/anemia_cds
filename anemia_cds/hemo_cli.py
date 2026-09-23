"""
hemo_cli.py — Command-line interface for HematoAI.

Usage:
  hemo run              Launch the full interactive terminal app (default)
  hemo train            Quick-launch: go straight to training
  hemo predict          Quick-launch: go straight to manual CBC entry
  hemo status           Show current model status without launching the UI
  hemo --help           Help

Install once with:
  pip install -e .

Then use from any directory:
  hemo run
"""
import os
import sys
import argparse

# ── Resolve anemia_cds root regardless of where hemo is called from ──────────
CLI_DIR = os.path.dirname(os.path.abspath(__file__))  # always anemia_cds/


def _enter_app_dir():
    """Change working directory to anemia_cds/ so all relative paths resolve."""
    os.chdir(CLI_DIR)
    if CLI_DIR not in sys.path:
        sys.path.insert(0, CLI_DIR)


# ─────────────────────────── Sub-commands ────────────────────────────────────

def cmd_run(_args):
    """Launch the full HematoAI interactive terminal."""
    _enter_app_dir()
    from main import main
    main()


def cmd_train(_args):
    """Jump directly into the Train/Retrain flow (menu option 6)."""
    _enter_app_dir()
    from rich.console import Console
    from modules.llm_client  import LLMClient
    from modules.data_loader import DataLoader
    import config, json

    console = Console()
    console.print("[bold cyan]HematoAI — Quick Train[/bold cyan]")

    # Load LLM client (same logic as main.py)
    backend = config.LLM_BACKEND
    if os.path.exists(config.SETTINGS_PATH):
        with open(config.SETTINGS_PATH) as f:
            backend = json.load(f).get('backend', backend)
    try:
        client = LLMClient(backend=backend)
    except Exception as e:
        console.print(f"[yellow]⚠ LLM: {e}[/yellow]")
        client = None

    from main import handle_train
    handle_train(DataLoader(), client)


def cmd_predict(_args):
    """Jump directly into the Manual CBC Predict flow (menu option 3)."""
    _enter_app_dir()
    from rich.console import Console
    from modules.predictor      import Predictor
    from modules.clinical_engine import ClinicalEngine
    from modules.llm_client     import LLMClient
    import config, json

    console = Console()
    console.print("[bold cyan]HematoAI — Quick Predict (Manual CBC)[/bold cyan]")

    backend = config.LLM_BACKEND
    if os.path.exists(config.SETTINGS_PATH):
        with open(config.SETTINGS_PATH) as f:
            backend = json.load(f).get('backend', backend)
    try:
        client = LLMClient(backend=backend)
        clinical = ClinicalEngine(client)
    except Exception as e:
        console.print(f"[yellow]⚠ LLM: {e}[/yellow]")
        clinical = None

    from main import handle_predict_manual
    handle_predict_manual(Predictor(), clinical, None)


def cmd_status(_args):
    """Print model + LLM status without launching the full UI."""
    _enter_app_dir()
    from rich.console import Console
    from rich.table   import Table
    from modules.llm_client import LLMClient
    from modules.trainer    import Trainer
    import config, json, os

    console = Console()

    # LLM status
    backend = config.LLM_BACKEND
    if os.path.exists(config.SETTINGS_PATH):
        with open(config.SETTINGS_PATH) as f:
            backend = json.load(f).get('backend', backend)
    try:
        client = LLMClient(backend=backend)
        llm_info = client.get_status()
        llm_ok   = True
    except Exception as e:
        llm_info = {'backend': backend, 'text_model': 'N/A', 'status': str(e)}
        llm_ok   = False

    # Model metadata
    meta = Trainer().load_metadata()

    table = Table(title="HematoAI Status", border_style="cyan", show_header=False)
    table.add_column("Key",   style="cyan", width=22)
    table.add_column("Value", style="white")

    llm_color = "green" if llm_ok else "red"
    table.add_row("LLM Backend",    f"[{llm_color}]{llm_info.get('backend','?')}[/{llm_color}]")
    table.add_row("Text Model",     llm_info.get('text_model', 'N/A'))
    table.add_row("LLM Status",     "[green]connected[/green]" if llm_ok else f"[red]{llm_info.get('status')}[/red]")

    if meta:
        table.add_row("─" * 20, "─" * 30)
        table.add_row("ML Model",    meta.get('model_name', 'N/A'))
        table.add_row("F1 Macro",    str(meta.get('f1_macro', 'N/A')))
        table.add_row("Trained At",  meta.get('trained_at', 'N/A'))
        ds = meta.get('dataset', {})
        table.add_row("Dataset",     f"{ds.get('source','N/A')} ({ds.get('rows','?')} rows)")
    else:
        table.add_row("ML Model",    "[yellow]Not trained yet — run: hemo train[/yellow]")

    model_exists = os.path.exists(config.MODEL_PATH)
    table.add_row("Model File",     "[green]✓ exists[/green]" if model_exists else "[red]✗ missing[/red]")

    console.print()
    console.print(table)
    console.print()


# ─────────────────────────── CLI Entry Point ─────────────────────────────────

def cli():
    parser = argparse.ArgumentParser(
        prog="hemo",
        description="HematoAI v3.0 — Explainable AI for Microcytic Anemia CDS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  hemo run          Launch the full interactive terminal app  [default]
  hemo train        Quick-launch: skip to model training
  hemo predict      Quick-launch: skip to manual CBC prediction
  hemo status       Show LLM + ML model status

Examples:
  hemo run
  hemo train
  hemo predict
  hemo status
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run",     help="Launch the full interactive terminal app (default)")
    subparsers.add_parser("train",   help="Quick-launch: go straight to training")
    subparsers.add_parser("predict", help="Quick-launch: go straight to manual CBC entry")
    subparsers.add_parser("status",  help="Show model + LLM status without launching the UI")

    args = parser.parse_args()

    dispatch = {
        "run":     cmd_run,
        "train":   cmd_train,
        "predict": cmd_predict,
        "status":  cmd_status,
        None:      cmd_run,   # `hemo` with no subcommand → same as `hemo run`
    }

    handler = dispatch.get(args.command, cmd_run)
    handler(args)


if __name__ == "__main__":
    cli()
