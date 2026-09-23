"""
clinical_engine.py — LLM-powered clinical report generation.
Uses LLMClient exclusively for all AI text generation.
Produces: EDA narrative, per-patient clinical explanation, action justification.
"""
from modules.llm_client import LLMClient
from rich.console import Console
from rich.panel import Panel
import config

console = Console()


class ClinicalEngine:
    """
    Generates clinical-grade text using LLM backend.
    All prompts mimic a hematologist's reasoning style.
    """

    def __init__(self, client: LLMClient):
        self.client = client

    # ─────────────────────────── EDA Narrative ───────────────────────────────

    def eda_narrative(self, stats_json: dict) -> str:
        """
        Generate a hematologist-style narrative explaining EDA findings.
        stats_json: per-class mean CBC statistics dict.
        Returns plain text with 4 bullet points.
        """
        prompt = (
            f"Given per-class CBC statistics: {stats_json}. "
            "As a hematologist, give 4 bullet points: "
            "(1) top discriminating parameters between IDA and Thalassemia Trait, "
            "(2) what RDW pattern reveals, "
            "(3) what RBC count pattern reveals, "
            "(4) dataset quality assessment. "
            "Plain language for a GP."
        )
        return self.client.chat(prompt, max_tokens=500)

    # ─────────────────────────── Per-Patient Clinical Report ─────────────────

    def clinical_explanation(
        self,
        cbc_values: dict,
        predicted_class: str,
        confidence: float,
        derived_indices: dict,
        shap_features: str,
    ) -> str:
        """
        Generate a 3-sentence clinical explanation for a single patient.
        Follows spec prompt exactly.
        """
        mentzer = derived_indices.get('Mentzer_Index', 0)
        shine_lal = derived_indices.get('Shine_Lal', 0)
        ef = derived_indices.get('England_Fraser', 0)

        mentzer_interp = 'IDA pattern' if mentzer > 13 else 'Thalassemia pattern'

        prompt = (
            f"Patient CBC: {cbc_values}. "
            f"ML prediction: {predicted_class} ({confidence:.1f}% confidence). "
            f"Derived indices: Mentzer={mentzer:.2f} ({mentzer_interp}), "
            f"Shine-Lal={shine_lal:.1f}, England-Fraser={ef:.2f}. "
            f"Top 5 SHAP features: {shap_features}. "
            "As a clinical hematologist, in exactly 3 sentences: "
            "(1) why this CBC pattern fits this diagnosis, "
            "(2) which single parameter is most diagnostic and why, "
            "(3) the most critical next step for a rural GP with limited resources. "
            "Be direct. No jargon."
        )
        return self.client.chat(prompt, temperature=0.2, max_tokens=350)

    # ─────────────────────────── Action Justification ────────────────────────

    def action_justification(
        self,
        predicted_class: str,
        confidence: float,
        cbc_values: dict,
    ) -> str:
        """
        Generate a brief action plan / clinical decision support recommendation.
        """
        prompt = (
            f"A patient with CBC {cbc_values} has been classified as "
            f"{predicted_class} with {confidence:.1f}% confidence by an ML model. "
            "As a hematologist advising a rural GP: "
            "In 2-3 sentences, state the most important actionable steps. "
            "Include: (1) whether immediate iron supplementation is appropriate, "
            "(2) key confirmatory test if affordable, "
            "(3) red flag requiring urgent specialist referral. "
            "Be specific. No jargon."
        )
        return self.client.chat(prompt, temperature=0.2, max_tokens=250)

    # ─────────────────────────── Display ─────────────────────────────────────

    def display_report(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: dict,
        cbc_values: dict,
        derived_indices: dict,
        explanation: str,
        action: str,
    ):
        """Render the complete clinical report as a rich terminal panel."""
        # Prediction header
        color = {
            'IDA': 'bold red',
            'Thalassemia_Trait': 'bold blue',
            'Anemia_of_Chronic_Disease': 'bold yellow',
        }.get(predicted_class, 'bold white')

        console.print()
        console.print(Panel(
            f"[{color}]🩺 Diagnosis: {predicted_class}[/{color}]\n"
            f"Confidence: [bold]{confidence:.1f}%[/bold]\n\n"
            f"Probabilities:\n" +
            "\n".join(f"  {cls}: {pct:.1f}%" for cls, pct in probabilities.items()),
            title="[bold]ML Prediction[/bold]",
            border_style="red"
        ))

        # Derived indices
        idx_lines = "\n".join(
            f"  {k}: [cyan]{v}[/cyan]"
            for k, v in derived_indices.items()
        )
        console.print(Panel(idx_lines, title="[bold]Derived Hematologic Indices[/bold]", border_style="blue"))

        # Clinical explanation
        console.print(Panel(
            explanation,
            title="[bold]🤖 AI Clinical Explanation[/bold]",
            border_style="green"
        ))

        # Action plan
        console.print(Panel(
            action,
            title="[bold]📋 Recommended Clinical Actions[/bold]",
            border_style="yellow"
        ))

    # ─────────────────────────── Export ────────────────────────────────────────

    def export_markdown_report(
        self,
        predicted_class: str,
        confidence: float,
        probabilities: dict,
        cbc_values: dict,
        derived_indices: dict,
        explanation: str,
        action: str,
    ) -> str:
        """Export the clinical report as a well-formatted markdown file."""
        import os
        from datetime import datetime

        os.makedirs(os.path.join(config.OUTPUT_DIR, 'reports'), exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Patient_Report_{timestamp}.md"
        filepath = os.path.join(config.OUTPUT_DIR, 'reports', filename)

        md_content = f"""# Hematology Clinical Decision Support Report
**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Diagnosis Prediction:** {predicted_class} ({confidence:.1f}% confidence)

## 📊 Probabilities
"""
        for cls, pct in probabilities.items():
            md_content += f"- **{cls}**: {pct:.1f}%\n"

        md_content += "\n## 🩸 Complete Blood Count (CBC) Values\n"
        md_content += "| Parameter | Value | Reference Range |\n"
        md_content += "|---|---|---|\n"
        for param, value in cbc_values.items():
            ref = config.BIOLOGIC_RANGES.get(param, ["N/A", "N/A"])
            md_content += f"| {param} | {value} | {ref[0]} - {ref[1]} |\n"

        md_content += "\n## 🔬 Derived Hematologic Indices\n"
        md_content += "| Index | Value |\n"
        md_content += "|---|---|\n"
        for index, value in derived_indices.items():
            md_content += f"| {index} | {value:.2f} |\n"

        md_content += f"\n## 🤖 AI Clinical Explanation\n\n{explanation}\n\n"
        md_content += f"## 📋 Recommended Clinical Actions\n\n{action}\n\n"
        md_content += "---\n*Generated by HematoAI v3.0 - Explainable AI for Microcytic Anemia CDS*"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(md_content)

        return filepath
