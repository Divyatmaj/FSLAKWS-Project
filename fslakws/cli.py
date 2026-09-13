
"""CLI interface for FSLAKWS using Python Fire."""

import fire

from rich.console import Console
from rich.table import Table

from . import detector


console = Console()


def _score_color(score: float) -> str:
    """Return a color based on confidence."""
    if score >= 0.8:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


class FSLAKWS:

    def detect(
        self,
        query_path: str = "query.wav",
        examples_dir: str = "examples",
        encoder_name: str = "base",
        language: str = "multi",
        device: str = "cpu",
        threshold: float = 0.5,
        hop_seconds: float = 0.25,
        smoothing_radius: int = 2,
        batch_size: int = 8,
        **keyword_examples,
    ):
        """Detect keywords using folder or ad-hoc examples."""
        keyword_paths = None

        if keyword_examples:
            keyword_paths = {
                kw: [p.strip() for p in paths.split(",")]
                for kw, paths in keyword_examples.items()
            }

        detections = detector.detect(
            query_path=query_path,
            examples_dir=examples_dir,
            keyword_paths=keyword_paths,
            encoder_name=encoder_name,
            language=language,
            device=device,
            threshold=threshold,
            hop_seconds=hop_seconds,
            smoothing_radius=smoothing_radius,
            batch_size=batch_size,
        )

        if not detections:
            console.print("[yellow]No detections above threshold.[/yellow]")
            return

        by_keyword: dict[str, list[detector.Detection]] = {}

        for detection in detections:
            by_keyword.setdefault(detection.keyword, []).append(detection)

        for keyword, hits in by_keyword.items():
            table = Table(
                title=f"Keyword: {keyword}",
                title_style="bold cyan",
            )

            table.add_column("Start (s)", justify="right")
            table.add_column("End (s)", justify="right")
            table.add_column("Score", justify="right")

            for hit in hits:
                color = _score_color(hit.score)

                table.add_row(
                    f"{hit.start:.1f}",
                    f"{hit.end:.1f}",
                    f"[{color}]{hit.score:.2f}[/{color}]",
                )

            console.print(table)

    def info(self):
        """Show package and device information."""
        import torch

        console.print("[bold cyan]fslakws V1[/bold cyan]")
        console.print(
            f"torch version: [green]{torch.__version__}[/green]"
        )

        mps = torch.backends.mps.is_available()

        console.print(
            f"MPS available: [green]{mps}[/green]"
            if mps
            else f"MPS available: [yellow]{mps}[/yellow]"
        )


def main():
    fire.Fire(FSLAKWS)


if __name__ == "__main__":
    main()
