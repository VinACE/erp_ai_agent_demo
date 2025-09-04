# -----------------------------
# Reconciliation Function
# -----------------------------
def reconcile(erp, bank):
    print("\n[INFO] 🔄 Starting reconciliation process...")

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

    print("[INFO] ✅ Reconciliation completed.")
    print("\n[DEBUG] Preview of reconciliation table:")
    print(merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]].head())

    # Save full reconciliation to CSV
    output_file = "reconciliation_report.csv"
    merged.to_csv(output_file, index=False)
    print(f"[INFO] Full reconciliation report saved to {output_file}")

    return merged[["Invoice ID", "Amount_erp", "Amount_bank", "Status_flag"]].to_dict(orient="records")


# -----------------------------
# Agent with LLM
# -----------------------------
def run_with_agent(erp_records, bank_records):
    print("\n[INFO] 🤖 Calling the LLM Agent for reconciliation query...")
    query = "Reconcile ERP and Bank transactions and summarize discrepancies."
    result = agent.run(query)
    print("[INFO] ✅ LLM Agent completed.")
    return result


# -----------------------------
# 4. Demo Runner
# -----------------------------
if __name__ == "__main__":
    print("🚀 Running ERP Agentic Demo...\n")

    erp_records = load_erp("erp_data.xlsx")
    bank_records = extract_bank_from_pdf("bank_statement.pdf")

    if erp_records and bank_records:
        # Direct reconciliation (CSV output)
        reconcile(erp_records, bank_records)

        # LLM-based reconciliation summary
        result = run_with_agent(erp_records, bank_records)
        print("\n✅ Final Result from LLM:\n", result)
    else:
        print("[INFO] Skipping agent execution because required files are missing.")
