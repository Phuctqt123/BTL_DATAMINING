"""Download UCI Online Retail II's original workbook (not for Git commits)."""
from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online%2Bretail%2Bii.zip"
FALLBACK_XLSX = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"


def _is_xlsx(path: Path) -> bool:
    """An XLSX is itself a ZIP archive containing xl/workbook.xml."""
    try:
        with zipfile.ZipFile(path) as zf:
            return "xl/workbook.xml" in zf.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def download(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and _is_xlsx(target):
        print(f"Workbook already exists: {target}")
        return target
    with tempfile.TemporaryDirectory(prefix="uci502_") as td:
        temp = Path(td)
        fetched = temp / "online_retail_II.xlsx"
        try:
            req = urllib.request.Request(UCI_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as response, (temp / "source.zip").open("wb") as out:
                shutil.copyfileobj(response, out)
            with zipfile.ZipFile(temp / "source.zip") as zf:
                names = [n for n in zf.namelist() if Path(n).name.lower() == "online_retail_ii.xlsx"]
                if len(names) != 1:
                    raise ValueError(f"Expected one UCI XLSX; found {zf.namelist()}")
                with zf.open(names[0]) as source, fetched.open("wb") as out:
                    shutil.copyfileobj(source, out)
        except (OSError, zipfile.BadZipFile, ValueError) as exc:
            print(f"Primary UCI archive failed ({exc}); trying legacy XLSX endpoint")
            req = urllib.request.Request(FALLBACK_XLSX, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as response, fetched.open("wb") as out:
                shutil.copyfileobj(response, out)
        if not _is_xlsx(fetched):
            raise ValueError("UCI download did not contain a valid XLSX workbook")
        shutil.move(str(fetched), str(target))
    print(f"Saved {target} ({target.stat().st_size / 1e6:.2f} MB)")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch UCI 502 workbook")
    parser.add_argument("--output", type=Path, default=Path("data/raw/online_retail_II.xlsx"))
    args = parser.parse_args()
    download(args.output)
