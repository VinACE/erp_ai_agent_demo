import pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# -----------------------------
# 1. Define Tools
# -----------------------------
def load_erp(file="erp_data.xlsx"):
    df = pd.read_excel(file)
    return df.to_dict(orient="records")

def load_bank(file="bank_statement.csv"):
    df = pd.read_csv(file)
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
        ("Amount mismatch" if r["Amount_erp"] != r["Amount_bank"] else "Match")
    ), axis=1)

    return merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]].to_dict(orient="records")

# -----------------------------
# 2. Set up Local LLM
# -----------------------------
llm_pipeline = pipeline(
    "text-generation",
    model="path/to/local/mixtral-7b",   # 👉 replace with your local model path
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

