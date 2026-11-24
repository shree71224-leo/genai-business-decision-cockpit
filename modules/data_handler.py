import pandas as pd


def read_dataset(upload):
    """Read CSV, XLS, or XLSX business datasets."""

    upload.seek(0)
    name = upload.name.lower()

    if name.endswith(".csv"):
        try:
            return pd.read_csv(upload)

        except UnicodeDecodeError:
            upload.seek(0)
            return pd.read_csv(upload, encoding="latin-1")

    if name.endswith(".xls"):
        return pd.read_excel(upload, engine="xlrd")

    if name.endswith(".xlsx"):
        return pd.read_excel(upload, engine="openpyxl")

    raise ValueError("Unsupported file format.")