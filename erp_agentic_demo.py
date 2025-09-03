import os
import pandas as pd
import pdfplumber
from langchain.agents import Tool, initialize_agent
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# -----------------------------
# 1. Define Tools
# -----------------------------
def load_erp(file="erp_data.xlsx"):
    if not os.path.exists(file):
        print(f"[ERROR] ERP file not found: {file}")
        return []
    df = pd.read_excel(file)
    print(f"[INFO] ERP records loaded: {len(df)}")
    return df.to_dict(orient="records")


def load_bank(pdf_file="bank_statement.pdf"):
    """Extract bank data directly from PDF and return as records"""
    if not os.path.exists(pdf_file):
        print(f"[ERROR] Bank PDF not found: {pdf_file}")
        return []

    rows = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            print(f"[DEBUG] Extracting page {i} from PDF...")
            text = page.extract_text()
            if text:
                print(f"[DEBUG] Page {i} text preview:\n{text[:300]}\n")
            table = page.extract_table()
            if table:
                headers = table[0]
                for row in table[1:]:
                    rows.append(dict(zip(headers, row)))

    if not rows:
        print("[WARN] No tables extracted from PDF. Check if it's image-based.")
        return []

    df = pd.DataFrame(rows)
    print(f"[INFO] Bank records extracted from PDF: {len(df)}")
    return df.to_dict(orient="records")


def reconcile(erp, bank):
    erp_df = pd.DataFrame(erp)
    bank_df = pd.DataFrame(bank)

    merged = pd.merge(
        erp_df, bank_df,
        how="outer",
        left_on="Invoice ID",
        right_on="Invoice ID",
        suffixes=("_erp", "_bank"),
        indicator=True
    )

    merged["Status_flag"] = merged.apply(lambda r: (
        "Missing in ERP" if r["_merge"] == "right_only" else
        "Missing in Bank" if r["_merge"] == "left_only" else
        ("Amount mismatch" if r.get("Amount_erp") != r.get("Amount_bank") else "Match")
    ), axis=1)

    print("\n[DEBUG] Intermediate reconciliation table:")
    print(merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]].head())

    return merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]].to_dict(orient="records")


# -----------------------------
# 2. Set up Local LLM
# -----------------------------
llm_pipeline = pipeline(
    "text-generation",
    model=os.path.expanduser("~/.cache/huggingface/hub/models--mistralai--mixtral-8x7b-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0"),
    device_map="auto",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# -----------------------------
# 3. Wrap Tools for Agent
# -----------------------------
tools = [
    Tool(name="Load ERP", func=load_erp, description="Load ERP transactions"),
    Tool(name="Load Bank", func=load_bank, description="Extract and load Bank transactions from PDF"),
    Tool(name="Reconcile", func=reconcile, description="Reconcile ERP and Bank data")
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# -----------------------------
# 4. Demo Runner
# -----------------------------
if __name__ == "__main__":
    print("🚀 Running ERP Agentic Demo...\n")

    erp_records = load_erp()
    bank_records = load_bank()

    if not erp_records or not bank_records:
        print("[INFO] Skipping agent execution because required data is missing.")
    else:
        query = "Reconcile ERP and Bank transactions and summarize discrepancies."
        result = agent.run(query)
        print("\n✅ Final Result:\n", result)
