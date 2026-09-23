"""
settings_manager.py — Runtime settings panel.
Switch LLM backend (Groq ↔ Ollama), change models, adjust temperature.
Settings persist across restarts via .settings.json.
No restart required when switching backends.
"""
import os
import json
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
import config

console = Console()

GROQ_TEXT_MODELS = [
    'llama-3.3-70b-versatile',
    'llama-3.1-8b-instant',
    'mixtral-8x7b-32768',
]

OLLAMA_SETUP_GUIDE = """
[bold yellow]━━━━━━━━━  Ollama Setup Guide  ━━━━━━━━━[/bold yellow]

[bold]Step 1: Install Ollama[/bold]
  Linux/Mac:  [cyan]curl -fsSL https://ollama.com/install.sh | sh[/cyan]
  Windows:    Download from [link=https://ollama.com/download]https://ollama.com/download[/link]

[bold]Step 2: Start the Ollama server[/bold]
  [cyan]ollama serve[/cyan]
  (Run in a separate terminal — keep it running)

[bold]Step 3: Pull recommended models[/bold]
  Text model:   [cyan]ollama pull llama3.2[/cyan]     (2.0 GB, 4 GB RAM)
  Vision model: [cyan]ollama pull llava[/cyan]         (4.7 GB, 8 GB RAM)

  Better reasoning (if RAM allows):
  [cyan]ollama pull llama3.1:8b[/cyan]                (4.7 GB, 8 GB RAM)

[bold]Step 4: Switch in HematoAI[/bold]
  Main Menu → S (Settings) → LLM Backend → ollama

[dim]With 16 GB RAM you can run llama3.1:8b + llava simultaneously.[/dim]
"""


class SettingsManager:
    """
    Manages runtime application settings.
    Saves to .settings.json for persistence across restarts.
    """

    def __init__(self, llm_client):
        self.client  = llm_client
        self.settings = self._load()

    # ─────────────────────────── Persistence ─────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(config.SETTINGS_PATH):
            try:
                with open(config.SETTINGS_PATH) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'backend':      config.LLM_BACKEND,
            'groq_model':   config.GROQ_TEXT_MODEL,
            'ollama_model': config.OLLAMA_TEXT_MODEL,
            'temperature':  config.DEFAULT_TEMPERATURE,
        }

    def _save(self):
        with open(config.SETTINGS_PATH, 'w') as f:
            json.dump(self.settings, f, indent=2)

    # ─────────────────────────── Main Settings Menu ───────────────────────────

    def show_settings_menu(self):
        """Display the settings panel and handle user choices."""
        while True:
            self._print_settings_panel()
            console.print()
            choice = Prompt.ask(
                "[bold cyan]Settings[/bold cyan] — Choose",
                choices=['1', '2', '3', 'b'],
                default='b'
            )

            if choice == '1':
                self._switch_backend()
            elif choice == '2':
                self._change_model()
            elif choice == '3':
                self._change_temperature()
            elif choice == 'b':
                break

    def _print_settings_panel(self):
        """Show current settings state."""
        status = self.client.get_status()

        table = Table(title="⚙ HematoAI Settings", border_style="yellow", show_header=True)
        table.add_column("Setting", style="cyan", width=25)
        table.add_column("Current Value", style="white")
        table.add_column("Options", style="dim")

        backend_color = 'green' if status['backend'].startswith('Groq') else 'yellow'
        table.add_row(
            "LLM Backend",
            f"[{backend_color}]{status['backend']}[/{backend_color}]",
            "groq / ollama"
        )
        table.add_row("Text Model",    status['text_model'],   "See option 2")
        table.add_row("Vision Model",  status['vision_model'], "Auto-selected")
        table.add_row(
            "Temperature",
            str(self.settings.get('temperature', 0.2)),
            "0.0 – 1.0"
        )

        if 'installed_models' in status:
            installed_str = ', '.join(status['installed_models']) or 'None'
            table.add_row("Installed Ollama Models", installed_str, "")

        console.print()
        console.print(table)
        console.print()
        console.print("  [bold cyan]1.[/bold cyan] Switch LLM Backend (Groq ↔ Ollama)")
        console.print("  [bold cyan]2.[/bold cyan] Change Model")
        console.print("  [bold cyan]3.[/bold cyan] Adjust Temperature")
        console.print("  [bold cyan]B.[/bold cyan] Back to Main Menu")

    # ─────────────────────────── Switch Backend ───────────────────────────────

    def _switch_backend(self):
        current = self.client.backend
        target  = 'ollama' if current == 'groq' else 'groq'

        console.print(f"\n  Switching from [yellow]{current}[/yellow] → [cyan]{target}[/cyan]")

        if target == 'ollama':
            self._switch_to_ollama()
        else:
            self._switch_to_groq()

    def _switch_to_groq(self):
        """Verify Groq key and perform switch."""
        api_key = os.getenv('GROQ_API_KEY', '').strip()
        if not api_key:
            console.print("[red]✗ GROQ_API_KEY not found in .env file.[/red]")
            console.print("  Add it: echo GROQ_API_KEY=your_key >> .env")
            console.print("  Get a free key at: https://console.groq.com")
            return

        console.print(f"  API key found. Text model: [cyan]{self.settings['groq_model']}[/cyan]")
        console.print(f"  Vision model: [cyan]{config.GROQ_VISION_MODEL}[/cyan]")

        if Confirm.ask("  Confirm switch to Groq?", default=True):
            try:
                self.client.set_backend('groq', self.settings['groq_model'])
                self.settings['backend'] = 'groq'
                self._save()
                console.print("[green]✓ Switched to Groq Cloud.[/green]")
            except Exception as e:
                console.print(f"[red]✗ Switch failed: {e}[/red]")

    def _switch_to_ollama(self):
        """Verify Ollama is running, show setup guide if not, perform switch."""
        console.print(Panel(OLLAMA_SETUP_GUIDE, border_style="yellow"))

        # Check if Ollama is running
        try:
            r = requests.get(f'{config.OLLAMA_BASE_URL}/api/tags', timeout=3)
            if r.status_code != 200:
                raise ConnectionError()
            installed = [m['name'] for m in r.json().get('models', [])]
        except Exception:
            console.print("[red]✗ Ollama is not running at localhost:11434.[/red]")
            console.print("  Run: [cyan]ollama serve[/cyan]  in a separate terminal, then try again.")
            return

        if not installed:
            console.print("[yellow]⚠ No Ollama models installed yet.[/yellow]")
            console.print("  Pull models: [cyan]ollama pull llama3.2[/cyan]  and  [cyan]ollama pull llava[/cyan]")
            return

        console.print(f"  Installed models: [cyan]{', '.join(installed)}[/cyan]")

        # Let user pick text model
        if len(installed) > 1:
            console.print("\n  Available text models:")
            for i, m in enumerate(installed, 1):
                console.print(f"    [cyan]{i}.[/cyan] {m}")
            idx = Prompt.ask("  Choose text model", choices=[str(i) for i in range(1, len(installed)+1)], default='1')
            selected_model = installed[int(idx) - 1]
        else:
            selected_model = installed[0]

        if Confirm.ask(f"  Switch to Ollama ({selected_model})?", default=True):
            try:
                self.client.set_backend('ollama', selected_model)
                self.settings['backend']      = 'ollama'
                self.settings['ollama_model'] = selected_model
                self._save()
                console.print(f"[green]✓ Switched to Ollama ({selected_model}).[/green]")
            except Exception as e:
                console.print(f"[red]✗ Switch failed: {e}[/red]")

    # ─────────────────────────── Change Model ────────────────────────────────

    def _change_model(self):
        if self.client.backend == 'groq':
            console.print("\n  Available Groq text models:")
            for i, m in enumerate(GROQ_TEXT_MODELS, 1):
                marker = " ★" if m == self.client.GROQ_TEXT_MODEL else ""
                console.print(f"    [cyan]{i}.[/cyan] {m}{marker}")
            idx = Prompt.ask("  Select model", choices=[str(i) for i in range(1, len(GROQ_TEXT_MODELS)+1)], default='1')
            selected = GROQ_TEXT_MODELS[int(idx) - 1]
            self.client.GROQ_TEXT_MODEL = selected
            self.settings['groq_model'] = selected
            self._save()
            console.print(f"[green]✓ Groq text model set to {selected}[/green]")

        else:  # ollama
            try:
                r = requests.get(f'{config.OLLAMA_BASE_URL}/api/tags', timeout=3)
                installed = [m['name'] for m in r.json().get('models', [])]
            except Exception:
                console.print("[red]Ollama not reachable.[/red]")
                return

            console.print("\n  Installed Ollama models:")
            for i, m in enumerate(installed, 1):
                marker = " ★" if m == self.client.OLLAMA_TEXT_MODEL else ""
                console.print(f"    [cyan]{i}.[/cyan] {m}{marker}")

            idx = Prompt.ask("  Select model", choices=[str(i) for i in range(1, len(installed)+1)], default='1')
            selected = installed[int(idx) - 1]
            self.client.OLLAMA_TEXT_MODEL = selected
            self.settings['ollama_model'] = selected
            self._save()
            console.print(f"[green]✓ Ollama text model set to {selected}[/green]")

    # ─────────────────────────── Temperature ─────────────────────────────────

    def _change_temperature(self):
        console.print(f"\n  Current temperature: [cyan]{self.settings.get('temperature', 0.2)}[/cyan]")
        console.print("  [dim]Recommended: 0.2 for clinical outputs. Range: 0.0 – 1.0[/dim]")
        raw = Prompt.ask("  New temperature [0.0–1.0]", default=str(self.settings.get('temperature', 0.2)))
        try:
            t = float(raw)
            t = max(0.0, min(1.0, t))
            self.settings['temperature'] = t
            self._save()
            console.print(f"[green]✓ Temperature set to {t}[/green]")
        except ValueError:
            console.print("[red]Invalid input.[/red]")
