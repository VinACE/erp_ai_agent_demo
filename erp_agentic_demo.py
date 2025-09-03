import pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import pprint

# -----------------------------
# Load ERP and Bank data
# -----------------------------
def load_erp_data():
    print("📂 Loading ERP data from erp_data.xlsx...")
    df = pd.read_excel("erp_data.xlsx")
    print("✅ ERP data sample:")
    print(df.head())  # show first few rows
    return df

def load_bank_data():
    print("📂 Loading Bank data from bank_statement.csv...")
    df = pd.read_csv("bank_statement.csv")
    print("✅ Bank data sample:")
    print(df.head())  # show first few rows
    return df

# -----------------------------
# Reconciliation tool
# -----------------------------
def reconcile():
    erp_df = load_erp_data()
    bank_df = load_bank_data()

    erp_df = erp_df.rename(columns={"Amount": "Amount"})
    bank_df = bank_df.rename(columns={"Debit/Credit": "Amount"})
    bank_df["Invoice ID"] = bank_df["Description"].str.extract(r"(INV-\d+)")

    merged = pd.merge(
        erp_df,
        bank_df,
        how="outer",
        on="Invoice ID",
        suffixes=("_erp", "_bank"),
        indicator=True
    )

    import math
    def amounts_match(a, b, tol=0.01):
        if pd.isna(a) or pd.isna(b):
            return False
        return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)

    merged["Status_flag"] = merged.apply(lambda row: (
        "Missing in ERP" if row["_merge"] == "right_only" else
        "Missing in Bank" if row["_merge"] == "left_only" else
        ("Amount mismatch" if not amounts_match(row["Amount_erp"], row["Amount_bank"]) else "Match")
    ), axis=1)

    # ✅ Debug print of reconciliation result
    print("\n🔍 Reconciliation table:")
    print(merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]])

    return merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]].to_dict(orient="records")

# -----------------------------
# Tools for the agent
# -----------------------------
tools = [
    Tool(name="Get ERP Data", func=lambda _: load_erp_data().to_dict(orient="records"), description="Load ERP data"),
    Tool(name="Get Bank Data", func=lambda _: load_bank_data().to_dict(orient="records"), description="Load Bank data"),
    Tool(name="Reconcile Transactions", func=lambda _: reconcile(), description="Match ERP and Bank transactions"),
]
