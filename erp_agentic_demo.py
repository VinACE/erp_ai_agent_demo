import os
import pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import pdfplumber
from datetime import datetime

# -----------------------------
# Helper for timestamped logs
# -----------------------------
def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} [{level}] {msg}")


# -----------------------------
# 1. Define Tools
# -----------------------------
def load_erp(file="erp_data.xlsx"):
    if not os.path.exists(file):
        log(f"ERP file not found: {file}", "ERROR")
        return []
    df = pd.read_excel(file)
    log(f"ERP records loaded: {len(df)}")
    return df.to_dict(orient="records")


def extract_bank_from_pdf(pdf_file="bank_statement.pdf"):
    if not os.path.exists(pdf_file):
        log(f"Bank PDF not found: {pdf_file}", "ERROR")
        return []

    records = []
    with pdfplumber.open(pdf_file) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            log(f"Extracted text from page {page_num}:\n{text}\n{'-'*40}", "DEBUG")
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

    log(f"Bank records extracted from PDF: {len(records)}")
    return records


def reconcile(erp, bank):
    log("🔄 Starting reconciliation process...")

    erp_df = pd.DataFrame(erp)
    bank_df = pd.DataFrame(bank)

    if erp_df.empty or bank_df.empty:
        log("One of the datasets is empty, skipping reconciliation.", "WARN")
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

    log("✅ Reconciliation completed.")

    log("Preview of reconciliation table:", "DEBUG")
    print(merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]].head())

    # Save full reconciliation to CSV
    output_file = "reconciliation_report.csv"
    merged.to_csv(output_file, index=False)
    log(f"Full reconciliation report saved to {output_file}")

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
# 4. Runner
# -----------------------------
def run_with_agent(erp_records, bank_records):
    log("🤖 Calling the LLM Agent for reconciliation query...")
    query = "Reconcile ERP and Bank transactions and summarize discrepancies."
    result = agent.run(query)
    log("✅ LLM Agent completed.")
    return result


if __name__ == "__main__":
    log("🚀 Running ERP Agentic Demo...\n")

    erp_records = load_erp("erp_data.xlsx")
    bank_records = extract_bank_from_pdf("bank_statement.pdf")

    if erp_records and bank_records:
        # Direct reconciliation (CSV output)
        reconcile(erp_records, bank_records)

        # LLM-based reconciliation summary
        result = run_with_agent(erp_records, bank_records)
        log("✅ Final Result from LLM:")
        print(result)
    else:
        log("Skipping agent execution because required files are missing.")
