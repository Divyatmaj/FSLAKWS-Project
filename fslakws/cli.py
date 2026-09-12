"""CLI entrypoint using python-fire."""

import fire

from . import detector


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
    ):
        """Detect keyword(s) in query_path using few-shot examples in examples_dir."""
        detections = detector.detect(
            examples_dir=examples_dir,
            query_path=query_path,
            encoder_name=encoder_name,
            language=language,
            device=device,
            threshold=threshold,
            hop_seconds=hop_seconds,
        )

        if not detections:
            print("No detections above threshold.")
            return

        by_keyword: dict[str, list[detector.Detection]] = {}
        for d in detections:
            by_keyword.setdefault(d.keyword, []).append(d)

        for keyword, hits in by_keyword.items():
            print(f"\nKeyword: {keyword}")
            print("Detected:")
            for h in hits:
                print(f"    {h.start:.1f} - {h.end:.1f} s   score={h.score:.2f}")

    def info(self):
        """Quick sanity check that the package + torch are importable."""
        import torch
        print("fslakws V1")
        print(f"torch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")


def main():
    fire.Fire(FSLAKWS)


if __name__ == "__main__":
    main()