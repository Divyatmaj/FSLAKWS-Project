"""
cli.py
------
Python Fire turns this class's methods into CLI commands automatically:

    python -m fslakws.cli detect --examples_dir=examples --query_path=audio.wav
    python -m fslakws.cli info
"""

import fire
from rich.console import Console
from rich.table import Table

from . import detector

console = Console()


def _score_color(score: float) -> str:
    # green/yellow/red by confidence band
    if score >= 0.8:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


class FSLAKWS:
    def detect(
        self,
        examples_dir: str,
        query_path: str,
        encoder_name: str = "base",
        language: str = "multi",
        device: str = "cpu",
        threshold: float = 0.5,
        hop_seconds: float = 0.25,
        smoothing_radius: int = 2,
    ):
        """
        Detect keyword(s) in query_path using few-shot examples in examples_dir.

        examples_dir layout:
            examples/computer/ex1.wav
            examples/computer/ex2.wav
            examples/hello/ex1.wav

        Folders named "negative" or "background" are used as contrast
        classes but never reported as detections themselves.
        """
        detections = detector.detect(
            examples_dir=examples_dir,
            query_path=query_path,
            encoder_name=encoder_name,
            language=language,
            device=device,
            threshold=threshold,
            hop_seconds=hop_seconds,
            smoothing_radius=smoothing_radius,
        )

        if not detections:
            console.print("[yellow]No detections above threshold.[/yellow]")
            return

        by_keyword: dict[str, list[detector.Detection]] = {}
        for d in detections:
            by_keyword.setdefault(d.keyword, []).append(d)

        for keyword, hits in by_keyword.items():
            table = Table(title=f"Keyword: {keyword}", title_style="bold cyan")
            table.add_column("Start (s)", justify="right")
            table.add_column("End (s)", justify="right")
            table.add_column("Score", justify="right")
            for h in hits:
                color = _score_color(h.score)
                table.add_row(f"{h.start:.1f}", f"{h.end:.1f}", f"[{color}]{h.score:.2f}[/{color}]")
            console.print(table)

    def info(self):
        """Quick sanity check that the package + torch are importable."""
        import torch
        console.print("[bold cyan]fslakws V1[/bold cyan]")
        console.print(f"torch version: [green]{torch.__version__}[/green]")
        mps = torch.backends.mps.is_available()
        console.print(f"MPS available: [green]{mps}[/green]" if mps else f"MPS available: [yellow]{mps}[/yellow]")


def main():
    fire.Fire(FSLAKWS)


if __name__ == "__main__":
    main()