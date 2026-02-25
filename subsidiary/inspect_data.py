## this script inspects the downloarded data itself
## check original dimensionality, columns, and what not

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"

# ── Load cached raw sample ─────────────────────────────────────────────
raw_path = DATA_DIR / "raw_sample.parquet"
if not raw_path.exists():
    print(f"❌  No cached data found at {raw_path}")
    print("   Run the Streamlit app first to fetch data, or run data/fetch.py directly.")
    sys.exit(1)

df = pd.read_parquet(raw_path)

# ── Basic shape & memory ──────────────────────────────────────────────
print("=" * 60)
print("RAW SAMPLE INSPECTION")
print("=" * 60)
print(f"\nShape:   {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
print(f"Memory:  {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

# ── Column listing with dtypes ────────────────────────────────────────
print("\n" + "-" * 60)
print("COLUMNS & DTYPES")
print("-" * 60)
for i, (col, dtype) in enumerate(df.dtypes.items(), 1):
    null_ct = df[col].isna().sum()
    null_pct = null_ct / len(df) * 100
    print(f"  {i:2d}. {col:<30s}  {str(dtype):<15s}  nulls: {null_ct:>5,} ({null_pct:.1f}%)")

# ── Numeric summary ──────────────────────────────────────────────────
print("\n" + "-" * 60)
print("NUMERIC SUMMARY (describe)")
print("-" * 60)
numeric_cols = df.select_dtypes(include="number").columns.tolist()
print(df[numeric_cols].describe().round(2).to_string())

# ── Categorical / object columns ──────────────────────────────────────
obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
if obj_cols:
    print("\n" + "-" * 60)
    print("CATEGORICAL / OBJECT COLUMNS")
    print("-" * 60)
    for col in obj_cols:
        unique = df[col].nunique()
        top = df[col].value_counts().head(5)
        print(f"\n  {col}  ({unique} unique values)")
        for val, ct in top.items():
            print(f"    {val!s:<30s}  {ct:>5,}")

# ── Datetime columns ─────────────────────────────────────────────────
dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
if dt_cols:
    print("\n" + "-" * 60)
    print("DATETIME COLUMNS")
    print("-" * 60)
    for col in dt_cols:
        print(f"\n  {col}")
        print(f"    min:  {df[col].min()}")
        print(f"    max:  {df[col].max()}")

# ── Sample rows ───────────────────────────────────────────────────────
print("\n" + "-" * 60)
print("FIRST 5 ROWS")
print("-" * 60)
print(df.head().to_string())

print("\n✅  Inspection complete.")
