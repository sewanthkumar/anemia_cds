"""
ui_components.py — All Rich panels, tables, banners, and menus for HematoAI.
Centralizes all terminal UI rendering to keep main.py clean.
"""
import os
import json
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.columns import Columns
import config

console = Console()

# ─────────────────────────── Header / Banner ──────────────────────────────────

LOGO = r"""
 ██╗  ██╗███████╗███╗   ███╗ █████╗ ████████╗ ██████╗  █████╗ ██╗
 ██║  ██║██╔════╝████╗ ████║██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗██║
 ███████║█████╗  ██╔████╔██║███████║   ██║   ██║   ██║███████║██║
 ██╔══██║██╔══╝  ██║╚██╔╝██║██╔══██║   ██║   ██║   ██║██╔══██║██║
 ██║  ██║███████╗██║ ╚═╝ ██║██║  ██║   ██║   ╚██████╔╝██║  ██║██║
 ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚═╝
                         AI v3.0  |  Microcytic Anemia CDS
"""


def print_header(llm_status: dict, model_metadata: dict):
    """Print the main application header panel."""
    backend_str = llm_status.get('backend', 'Unknown')
    text_model  = llm_status.get('text_model', 'N/A')

    md = model_metadata or {}
    dataset_info = md.get('dataset', {})
    dataset_name = dataset_info.get('source', 'None')
    dataset_rows = dataset_info.get('rows', 0)
    ml_model     = md.get('model_name', 'Not trained')
    f1           = md.get('f1_macro', 0.0)
    trained_at   = md.get('trained_at', 'N/A')

    backend_color = 'green' if 'Groq' in backend_str else 'yellow'

    header_text = (
        f"[bold white]HematoAI v3.0[/bold white]  |  [dim]Explainable AI — Microcytic Anemia CDS[/dim]\n"
        f"LLM Backend: [{backend_color}]{backend_str}[/{backend_color}]  |  Model: [cyan]{text_model}[/cyan]\n"
        f"Active Dataset: [white]{dataset_name} ({dataset_rows} rows)[/white]  |  "
        f"ML Model: [cyan]{ml_model}[/cyan]  |  F1: [bold green]{f1}[/bold green]  |  Trained: [dim]{trained_at}[/dim]"
    )

    console.print(LOGO, style="bold red", highlight=False)
    console.print(Panel(header_text, border_style="bold red", padding=(0, 2)))


def print_menu():
    """Print the main navigation menu."""
    menu = Table.grid(padding=(0, 4))
    menu.add_column(style="bold cyan", width=4)
    menu.add_column(style="white")

    items = [
        ("1.", "Predict from PDF Lab Report"),
        ("2.", "Predict from Image / Photo of Report"),
        ("3.", "Enter CBC Values Manually"),
        ("4.", "Predict Batch from CSV"),
        ("──", "──────────────────────────────────"),
        ("5.", "Run EDA (Exploratory Data Analysis)"),
        ("6.", "Train / Retrain Model"),
        ("7.", "Load Real Dataset"),
        ("──", "──────────────────────────────────"),
        ("8.", "View Model Performance"),
        ("9.", "Feature Importance (SHAP)"),
        ("──", "──────────────────────────────────"),
        ("S.", "Settings  (Switch Groq ↔ Ollama)"),
        ("Q.", "Quit"),
    ]
    for key, label in items:
        if key == "──":
            menu.add_row("[dim]──[/dim]", f"[dim]{label}[/dim]")
        else:
            menu.add_row(f"[bold cyan]{key}[/bold cyan]", label)

    console.print()
    console.print(Panel(menu, title="[bold]Main Menu[/bold]", border_style="blue", padding=(1, 4)))


# ─────────────────────────── Manual CBC Input ────────────────────────────────

def prompt_cbc_manual() -> dict:
    """Interactively prompt user to enter CBC values from keyboard."""
    from rich.prompt import FloatPrompt

    console.print(Panel(
        "[bold]Enter CBC values for prediction.[/bold]\n"
        "[dim]Press Enter to skip (not recommended — affects accuracy).[/dim]",
        border_style="cyan", title="Manual CBC Entry"
    ))

    cbc = {}
    for feat, (lo, hi) in config.BIOLOGIC_RANGES.items():
        while True:
            try:
                raw = console.input(f"  [cyan]{feat}[/cyan] [{lo}–{hi}]: ")
                if raw.strip() == '':
                    cbc[feat] = None
                    break
                val = float(raw)
                if not (lo <= val <= hi):
                    console.print(f"  [yellow]⚠ Value {val} is outside biologic range [{lo}, {hi}][/yellow]")
                cbc[feat] = val
                break
            except ValueError:
                console.print("  [red]Invalid input — please enter a number.[/red]")

    missing = [k for k, v in cbc.items() if v is None]
    if missing:
        console.print(f"[yellow]⚠ Missing values: {missing} — will use dataset medians during prediction.[/yellow]")

    return cbc


# ─────────────────────────── Model Performance ───────────────────────────────

def print_model_performance(metadata: dict, results: dict = None):
    """Display saved model performance metrics."""
    if not metadata:
        console.print("[yellow]No trained model found. Run option 6 first.[/yellow]")
        return

    table = Table(title="Current Model Performance", border_style="green")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Model Name",     metadata.get('model_name', 'N/A'))
    table.add_row("F1 Macro",       str(metadata.get('f1_macro', 'N/A')))
    table.add_row("Accuracy",       str(metadata.get('accuracy', 'N/A')))
    table.add_row("Trained At",     metadata.get('trained_at', 'N/A'))
    table.add_row("Features",       str(len(metadata.get('features', []))))

    ds = metadata.get('dataset', {})
    table.add_row("Dataset Source", ds.get('source', 'N/A'))
    table.add_row("Dataset Rows",   str(ds.get('rows', 'N/A')))

    console.print(table)


# ─────────────────────────── EDA Plots ───────────────────────────────────────

def run_eda_plots(df, narrative: str = None):
    """Generate and save EDA plots. Display narrative if provided."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os

    os.makedirs(os.path.join(config.OUTPUT_DIR, 'eda'), exist_ok=True)

    # Distribution plots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('CBC Feature Distributions by Class', fontsize=14, fontweight='bold')
    palette = {'IDA': '#e74c3c', 'Thalassemia_Trait': '#3498db', 'Anemia_of_Chronic_Disease': '#f39c12'}

    for i, feat in enumerate(config.RAW_FEATURES):
        ax = axes[i // 4][i % 4]
        if feat in df.columns:
            for label, color in palette.items():
                subset = df[df['label'] == label][feat].dropna()
                ax.hist(subset, bins=30, alpha=0.6, label=label, color=color)
            ax.set_title(feat)
            ax.set_xlabel(feat)
            ax.legend(fontsize=6)

    # Hide last empty cell
    axes[1][3].set_visible(False)
    plt.tight_layout()
    dist_path = os.path.join(config.OUTPUT_DIR, 'eda', 'cbc_distributions.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df[config.RAW_FEATURES].dropna()
    sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
    ax.set_title('CBC Feature Correlation Heatmap')
    corr_path = os.path.join(config.OUTPUT_DIR, 'eda', 'correlation_heatmap.png')
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    plt.close()

    console.print(f"[green]✓ EDA plots saved → {os.path.join(config.OUTPUT_DIR, 'eda')}[/green]")
    console.print(f"  • {dist_path}")
    console.print(f"  • {corr_path}")

    if narrative:
        console.print(Panel(narrative, title="[bold]🤖 Hematologist Narrative (AI)[/bold]", border_style="green"))

    return df['label'].value_counts().to_dict()


# ─────────────────────────── Batch Results ───────────────────────────────────

def print_batch_results(df):
    """Display batch prediction results as a Rich table."""
    table = Table(title=f"Batch Predictions ({len(df)} patients)", border_style="cyan")
    cols = ['predicted_class', 'confidence_pct'] + list(config.TARGET_CLASSES)
    base_cols = [c for c in df.columns if c not in cols + config.RAW_FEATURES]

    for col in (base_cols[:2] + cols):
        if col in df.columns:
            table.add_column(col, style="white")

    color_map = {
        'IDA': 'red',
        'Thalassemia_Trait': 'blue',
        'Anemia_of_Chronic_Disease': 'yellow'
    }
    for _, row in df.iterrows():
        cls = str(row.get('predicted_class', ''))
        color = color_map.get(cls, 'white')
        vals = []
        for col in (base_cols[:2] + cols):
            if col in df.columns:
                v = row[col]
                if col == 'predicted_class':
                    vals.append(f"[{color}]{v}[/{color}]")
                elif isinstance(v, float):
                    vals.append(f"{v:.1f}")
                else:
                    vals.append(str(v))
        table.add_row(*vals)

    console.print(table)


# ─────────────────────────── Boot Animation ────────────────────────────────────

def play_boot_animation():
    """Jarvis-style boot animation with Blood and Data themes."""
    from rich.live import Live
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
    from rich.console import Group
    from rich.panel import Panel
    import time
    import random

    console.clear()
    
    progress = Progress(
        SpinnerColumn(spinner_name="point", style="bold red"),
        TextColumn("[bold red]{task.description}"),
        BarColumn(bar_width=40, style="dark_red", complete_style="red"),
        TextColumn("[bold cyan]{task.percentage:>3.0f}%"),
    )
    
    task1 = progress.add_task("Initializing Hemo-Core...", total=100)
    task2 = progress.add_task("Loading Erythrocyte Data...", total=100)
    task3 = progress.add_task("Calibrating LLM Diagnostics...", total=100)
    
    def generate_data_stream():
        return "\n".join(" ".join(f"{random.randint(0, 255):02X}" for _ in range(12)) for _ in range(5))

    with Live(refresh_per_second=20) as live:
        for i in range(101):
            progress.update(task1, completed=i)
            if i > 20:
                progress.update(task2, completed=(i-20)*1.25)
            if i > 50:
                progress.update(task3, completed=(i-50)*2)
            
            data_panel = Panel(
                f"[dim cyan]{generate_data_stream()}[/dim cyan]",
                title="[bold red]BIO-METRICS STREAM[/bold red]",
                border_style="red"
            )
            
            main_group = Group(
                Panel(Align.center("[bold red]HEMETO-AI // JARVIS PROTOCOL INITIALIZING[/bold red]", vertical="middle"), border_style="dark_red"),
                Panel(progress, border_style="red"),
                data_panel
            )
            live.update(main_group)
            time.sleep(0.02)
        time.sleep(0.5)
    console.clear()


def run_with_jarvis_spinner(task_msg, func, *args, **kwargs):
    """Runs a function in a thread while showing a JARVIS-style blood/data animation."""
    from rich.live import Live
    from rich.panel import Panel
    from rich.console import Group
    from rich.align import Align
    import time
    import random
    import concurrent.futures
    
    def generate_data_stream():
        return "\n".join(" ".join(f"{random.randint(0, 255):02X}" for _ in range(12)) for _ in range(3))

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        
        with Live(refresh_per_second=20) as live:
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            i = 0
            while not future.done():
                frame = frames[i % len(frames)]
                i += 1
                
                data_str = f"[dim cyan]{generate_data_stream()}[/dim cyan]"
                
                msg_panel = Panel(f"[bold red]{frame} {task_msg}[/bold red]", border_style="dark_red")
                data_panel = Panel(data_str, border_style="red", title="[bold red]NEURAL LINK ACTIVE[/bold red]")
                
                live.update(Group(msg_panel, data_panel))
                time.sleep(0.05)
                
        return future.result()
