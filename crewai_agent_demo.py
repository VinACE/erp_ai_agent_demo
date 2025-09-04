# erp_reconcile_crewai.py

from crewai import Agent, Task, Crew
import pandas as pd
import pdfplumber

# --- File readers ---

def load_erp(file_path="erp_data.xlsx"):
    try:
        return pd.read_excel(file_path)
    except FileNotFoundError:
        print(f"[ERROR] ERP file not found: {file_path}")
        return pd.DataFrame()

def load_bank(file_path="bank_statement.pdf"):
    try:
        records = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for line in text.splitlines():
                        parts = line.split()
                        if len(parts) >= 3 and parts[-1].replace(".", "", 1).isdigit():
                            records.append({
                                "Date": parts[0],
                                "Description": " ".join(parts[1:-1]),
                                "Amount": float(parts[-1])
                            })
        return pd.DataFrame(records)
    except FileNotFoundError:
        print(f"[ERROR] Bank file not found: {file_path}")
        return pd.DataFrame()

def reconcile_records(erp_df, bank_df):
    merged = pd.merge(
        erp_df, bank_df,
        on=["Date", "Amount"], how="outer", indicator=True
    )
    return merged

# --- Define Agents ---

data_loader = Agent(
    role="Data Loader",
    goal="Load ERP and Bank records from Excel and PDF",
    backstory="You are responsible for making sure financial data is available for reconciliation.",
)

reconciler = Agent(
    role="Reconciler",
    goal="Compare ERP and Bank records to identify mismatches",
    backstory="You are an expert at spotting reconciliation differences between ledgers and statements.",
)

summarizer = Agent(
    role="Summarizer",
    goal="Generate a human-readable summary of reconciliation results",
    backstory="You explain reconciliation outcomes clearly for finance managers.",
)

# --- Define Tasks ---

task_load = Task(
    description="Load ERP Excel and Bank PDF data.",
    agent=data_loader,
    expected_output="Two dataframes containing ERP and Bank records.",
    function=lambda: (load_erp(), load_bank())
)

task_reconcile = Task(
    description="Run reconciliation between ERP and Bank data.",
    agent=reconciler,
    expected_output="A dataframe with matched and mismatched transactions.",
    function=lambda: reconcile_records(load_erp(), load_bank())
)

task_summary = Task(
    description="Summarize reconciliation results in plain language.",
    agent=summarizer,
    expected_output="A natural language summary highlighting matches and mismatches.",
    function=lambda: f"Reconciliation complete. Report saved to reconciliation_report.csv"
)

# --- Crew Orchestration ---

crew = Crew(
    agents=[data_loader, reconciler, summarizer],
    tasks=[task_load, task_reconcile, task_summary],
    verbose=True
)

if __name__ == "__main__":
    print("🚀 Running ERP Reconciliation with CrewAI...\n")
    result = crew.kickoff()

    # Save reconciliation report
    erp_df = load_erp()
    bank_df = load_bank()
    merged = reconcile_records(erp_df, bank_df)
    merged.to_csv("reconciliation_report.csv", index=False)

    print("\n✅ Reconciliation complete. Results saved in reconciliation_report.csv")
    print("CrewAI Result:", result)
