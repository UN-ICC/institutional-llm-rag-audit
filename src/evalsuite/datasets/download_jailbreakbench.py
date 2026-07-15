"""Download JailbreakBench dataset from HuggingFace.

Source: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
Citation: Chao et al., "JailbreakBench: An Open Robustness Benchmark
          for Jailbreaking Large Language Models", 2024.

Run: python -m evalsuite.datasets.download_jailbreakbench
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/benchmarks")


def main():
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading JailbreakBench from HuggingFace...")
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")

    # Harmful behaviors
    harmful = ds["harmful"]
    df_harmful = pd.DataFrame(harmful)
    out_path = DATA_DIR / "jailbreakbench_harmful.csv"
    df_harmful.to_csv(out_path, index=False)
    print(f"  Saved {len(df_harmful)} harmful behaviors to {out_path}")

    # Benign behaviors (for overrefusal layer)
    if "benign" in ds:
        benign = ds["benign"]
        df_benign = pd.DataFrame(benign)
        out_path = DATA_DIR / "jailbreakbench_benign.csv"
        df_benign.to_csv(out_path, index=False)
        print(f"  Saved {len(df_benign)} benign behaviors to {out_path}")


if __name__ == "__main__":
    main()
