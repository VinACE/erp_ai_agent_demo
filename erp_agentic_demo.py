import os
import pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import pdfplumber

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


def extract_bank_from_pdf(pdf_file="bank_statement.pdf"):
    if not os.path.exists(pdf_file):
        print(f"[ERROR] Bank PDF not found: {pdf_file}")
        return []

    records = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            print(f"[DEBUG] Extracted text from page {page_num}:\n{text}\n{'-'*40}")
            if not text:
                continue
            lines = text.split("\n")
            for line in lines:
                parts = line.split()
                # Expected format: Date | Description | Invoice | Amount | RefID
                if len(parts) >= 5 and parts[1] == "Payment":
                    try:
                        invoice_id = parts[2]       # e.g. INV0001
                        amount = float(parts[3])    # e.g. 267.1
                        records.append({"Invoice ID": invoice_id, "Amount": amount})
                    except ValueError:
                        continue

    print(f"[INFO] Bank records extracted from PDF: {len(records)}")
    return records


def reconcile(erp, bank):
    erp_df = pd.DataFrame(erp)
    bank_df = pd.DataFrame(bank)

    if erp_df.empty or bank_df.empty:
        print("[WARN] One of the datasets is empty, skipping reconciliation.")
        return []

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
        ("Amount mismatch" if r["Amount_erp"] != r["Amount_bank"] else "Match")
    ), axis=1)

    # Print preview
    print("\n[DEBUG] Preview of reconciliation table:")
    print(merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]].head())

    # Save full reconciliation to CSV
    output_file = "reconciliation_report.csv"
    merged.to_csv(output_file, index=False)
    print(f"[INFO] Full reconciliation report saved to {output_file}")

    return merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]].to_dict(orient="records")


# -----------------------------
# 2. Set up Local LLM
# -----------------------------
MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--mistralai--mixtral-8x7b-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0"
)

llm_pipeline = pipeline(
    "text-generation",
    model=MODEL_PATH,
    device_map="auto",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# -----------------------------
# 3. Wrap Tools for Agent
# -----------------------------
tools = [
    Tool(name="Load ERP", func=load_erp, description="Load ERP transactions"),
    Tool(name="Load Bank", func=extract_bank_from_pdf, description="Extract Bank transactions from PDF"),
    Tool(name="Reconcile", func=reconcile, description="Reconcile ERP and Bank data")
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# -----------------------------
# 4. Demo Runner
# -----------------------------
if __name__ == "__main__":
    print("🚀 Running ERP Agentic Demo...\n")

    erp_records = load_erp("erp_data.xlsx")
    bank_records = extract_bank_from_pdf("bank_statement.pdf")

    if erp_records and bank_records:
        query = "Reconcile ERP and Bank transactions and summarize discrepancies."
        result = agent.run(query)
        print("\n✅ Final Result:\n", result)
    else:
        print("[INFO] Skipping agent execution because required files are missing.")
