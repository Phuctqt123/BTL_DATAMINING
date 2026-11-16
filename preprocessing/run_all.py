"""One command to generate RFM, plots, and chapters using one source read.

    python -m preprocessing.run_all --input data/raw/online_retail_II.xlsx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from .preprocess import load_retail_data, run_pipeline_from_frame
from .eda_utils import make_eda_outputs
#from reports.generate_report import generate_report


def run_all(input_path: Path, root: Path) -> None:
    root = root.resolve()
    raw = load_retail_data(input_path)
    metadata = run_pipeline_from_frame(raw, root)
    features = pd.read_csv(
        root / "data/processed/customers_features.csv", index_col="Customer ID"
    )
    features.index = features.index.astype(str)
    features.index.name = "Customer ID"
    observations = make_eda_outputs(raw, features, root)
    #report = generate_report(root)
    print(json.dumps({
        "source_rows": metadata["raw_rows"],
        "customers": metadata["customers_with_purchases"],
        "peak_sales_month": observations["monthly_peak_sales"],
        #"report": str(report),
        "charts_dir": str(root / "preprocessing/figures"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run whole TV1 preprocessing + EDA + report")
    parser.add_argument("--input", type=Path, default=Path("data/raw/online_retail_II.xlsx"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run_all(args.input, args.root)
