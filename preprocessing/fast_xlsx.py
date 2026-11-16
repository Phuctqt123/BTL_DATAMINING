"""Streaming OOXML reader for the 1.07M-row original UCI 502 workbook.

It reads the two explicitly required worksheets without openpyxl's expensive
per-cell object construction, then returns the same schema as pandas.read_excel.
There are no Excel formulas in the UCI transaction source; this module reads
stored numeric/string cell values only.
"""
from __future__ import annotations

import csv
import re
from html import unescape
import tempfile
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from lxml import etree

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _shared_strings(zf: ZipFile) -> list[str]:
    """Read shared strings, including rich-text cells with multiple <t> nodes."""
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    out: list[str] = []
    with zf.open("xl/sharedStrings.xml") as file:
        for _, node in etree.iterparse(file, events=("end",), tag=NS + "si"):
            out.append("".join((t.text or "") for t in node.iter(NS + "t")))
            node.clear()
            while node.getprevious() is not None:
                del node.getparent()[0]
    return out


def _col_index(cell_ref: str) -> int:
    n = 0
    for ch in cell_ref:
        if not ch.isalpha():
            break
        n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
    return n - 1


# The official UCI workbook uses simple OOXML rows with <c r="..."> and
# <v> nodes. A compiled byte-regex is far faster than building 8M XML cells.
# The header and mandatory row counts are validated after parsing.
ROW_PATTERN = re.compile(rb'<row\b[^>]*>(.*?)</row>', re.DOTALL)
CELL_PATTERN = re.compile(rb'<c r="([A-H])\d+"([^>]*)>(.*?)</c>', re.DOTALL)
VALUE_PATTERN = re.compile(rb'<v>([^<]*)</v>')
INLINE_PATTERN = re.compile(rb'<t[^>]*>(.*?)</t>', re.DOTALL)


def _write_sheet(zf: ZipFile, member: str, name: str, strings: list[str], writer: csv.writer) -> int:
    expected_cols = ["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"]
    header_found = False
    nrows = 0
    xml_bytes = zf.read(member)
    for row_match in ROW_PATTERN.finditer(xml_bytes):
        cells: list[str | None] = [None] * 8
        for cell_match in CELL_PATTERN.finditer(row_match.group(1)):
            ix = cell_match.group(1)[0] - 65  # columns A-H, zero-indexed
            attrs = cell_match.group(2)
            body = cell_match.group(3)
            if b't="inlineStr"' in attrs:
                cells[ix] = unescape("".join(m.group(1).decode("utf-8") for m in INLINE_PATTERN.finditer(body)))
                continue
            value_match = VALUE_PATTERN.search(body)
            if value_match is None:
                continue
            raw = value_match.group(1)
            if b't="s"' in attrs:
                cells[ix] = strings[int(raw)]
            else:
                cells[ix] = raw.decode("utf-8")
        if not header_found:
            norm_header = [str(item or "").strip().lower().replace(" ", "") for item in cells]
            norm_expected = [col.lower().replace(" ", "") for col in expected_cols]
            if norm_header != norm_expected:
                raise ValueError(f"Unexpected columns in {name}: {cells}")
            header_found = True
        else:
            writer.writerow(cells + [name])
            nrows += 1
    del xml_bytes
    if not header_found or nrows < 1:
        raise ValueError(f"Missing or empty worksheet: {name}")
    return nrows


def load_retail_xlsx_fast(workbook: str | Path) -> pd.DataFrame:
    """Read the original UCI XLSX in streaming mode, both mandatory sheets."""
    workbook = Path(workbook)
    if not workbook.is_file():
        raise FileNotFoundError(workbook)
    members = {
        "Year 2009-2010": "xl/worksheets/sheet1.xml",
        "Year 2010-2011": "xl/worksheets/sheet2.xml",
    }
    with tempfile.TemporaryDirectory(prefix="uci502_fast_") as td:
        csv_path = Path(td) / "transactions.csv"
        with ZipFile(workbook) as zf:
            present = set(zf.namelist())
            if any(member not in present for member in members.values()):
                raise ValueError("Workbook is not the expected two-sheet UCI Online Retail II source")
            strings = _shared_strings(zf)
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country", "SourceSheet"])
                for sheet_name, member in members.items():
                    nrows = _write_sheet(zf, member, sheet_name, strings, writer)
                    print(f"[fast-xlsx] parsed {sheet_name}: {nrows:,} rows", flush=True)
        df = pd.read_csv(
            csv_path, dtype={
                "Invoice": "string", "StockCode": "string", "Description": "string",
                "Customer ID": "string", "Country": "string", "SourceSheet": "string",
            }, keep_default_na=False, na_values=[""], low_memory=False,
        )
    # Excel stores date-times as fractional days since the 1900-date epoch.
    numeric_dates = pd.to_numeric(df["InvoiceDate"], errors="coerce")
    timestamp = pd.to_datetime(numeric_dates, unit="D", origin="1899-12-30", errors="coerce").dt.round("s")
    if numeric_dates.isna().any():
        positions = numeric_dates.isna()
        timestamp.loc[positions] = pd.to_datetime(df.loc[positions, "InvoiceDate"], errors="coerce")
    df["InvoiceDate"] = timestamp
    for key in ("Invoice", "StockCode", "Customer ID"):
        df[key] = df[key].str.strip().str.replace(r"\.0+$", "", regex=True)
    for key in ("Country", "Description"):
        df[key] = df[key].str.strip().replace("", pd.NA)
    for key in ("Quantity", "Price"):
        df[key] = pd.to_numeric(df[key], errors="coerce")
    return df
