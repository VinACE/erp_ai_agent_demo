import os
import pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# -----------------------------
# 1. Define Tools with Exception Handling
# -----------------------------
def load_erp(file="erp_data.xlsx"):
    if not os.path.exists(file):
        print(f"[ERROR] ERP file not found: {file}")
        return []
    try:
        df = pd.read_excel(file)
        print(f"[INFO] Loaded ERP data from {file}, rows={len(df)}")
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"[ERROR] Failed to read ERP file: {e}")
        return []

def load_bank(file="bank_statement.csv"):
    if not os.path.exists(file):
        print(f"[ERROR] Bank file not found: {file}")
        return []
    try:
        df = pd.read_csv(file)
        print(f"[INFO] Loaded Bank data from {file}, rows={len(df)}")
        return df.to_dict(orient="records")
    except Exception as e:
        print(f"[ERROR] Failed to read Bank file: {e}")
        return []

def reconcile(erp, bank):
    if not erp or not bank:
        print("[WARN] Missing ERP or Bank data, cannot reconcile.")
        return [{"Invoice ID": None, "Amount_erp": None, "Amount_bank": None, "Status_flag": "No Data"}]

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
        ("Amount mismatch" if r["Amount_erp"] != r["Amount_bank"] else "Match")
    ), axis=1)

    print("[DEBUG] Reconciliation Table:")
    print(merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]])

    return merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]].to_dict(orient="records")

# -----------------------------
# 2. Set up Local LLM
# -----------------------------
llm_pipeline = pipeline(
    "text-generation",
    model="/home/vinsentparamanantham/.cache/huggingface/hub/models--mistralai--mixtral-8x7b-v0.1/snapshots/fc7ac94680e38d7348cfa806e51218e6273104b0",
    device_map="auto",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# -----------------------------
# 3. Wrap Tools for Agent
# -----------------------------
tools = [
    Tool(name="Load ERP", func=load_erp, description="Load ERP transactions"),
    Tool(name="Load Bank", func=load_bank, description="Load Bank transactions"),
    Tool(name="Reconcile", func=reconcile, description="Reconcile ERP and Bank data")
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# -----------------------------
# 4. Demo Runner
# -----------------------------
if __name__ == "__main__":
    print("🚀 Running ERP Agentic Demo...\n")
    query = "Reconcile ERP and Bank transactions and summarize discrepancies."
    result = agent.run(query)
    print("\n✅ Final Result:\n", result)
