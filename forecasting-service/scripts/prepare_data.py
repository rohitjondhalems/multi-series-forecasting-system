"""
Step-by-step data preparation.

Usage:
    python -m scripts.prepare_data          # read, check, and prepare real data
    python -m scripts.prepare_data --sample # generate sample data for testing
"""
import argparse
import glob
import numpy as np
import pandas as pd

from src import config as C


# ──────────────────────────────────────────────
# STEP 1: Read all CSV files into one dataframe
# ──────────────────────────────────────────────
def read_data():
    """Read all CSVs from data/raw/ into a single dataframe."""
    files = sorted(glob.glob(str(C.DATA_RAW / "*.csv")))

    if not files:
        print(f"ERROR: No CSV files found in {C.DATA_RAW}")
        print("Put the OneDrive files there, or run: python -m scripts.prepare_data --sample")
        return None

    print(f"Found {len(files)} files:")
    for f in files:
        print(f"  - {f}")

    # detect delimiter (tab or comma) from the first file
    with open(files[0], "r", encoding="utf-8-sig") as fh:
        header = fh.readline()
    sep = "\t" if header.count("\t") >= header.count(",") else ","
    sep_name = "TAB" if sep == "\t" else "COMMA"
    print(f"\nDelimiter: {sep_name}")

    # read and combine all files
    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f, sep=sep))
    df = pd.concat(dfs, ignore_index=True)

    print(f"\nTotal rows: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    return df


# ──────────────────────────────────────────────
# STEP 2: Check the data (schema + quality)
# ──────────────────────────────────────────────
def check_data(df):
    """Validate schema and print data quality summary."""
    print("\n" + "=" * 50)
    print("DATA QUALITY CHECK")
    print("=" * 50)

    # check required columns exist
    required = [C.TIME_COL, C.ID_COL, C.TARGET] + C.COV_NUM_COLS + [C.COV_CAT_COL]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"FAIL: Missing columns: {missing_cols}")
        return False
    print("PASS: All required columns present")

    # how many series?
    series = df[C.ID_COL].unique()
    print(f"\nSeries found ({len(series)}): {list(series)}")
    
    describe_d = df.describe(include="all").transpose()
    print(f"\n Stats Description, {describe_d}")

    # time range per series
    df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL])
    print("\nTime ranges:")
    for sid in series:
        s = df[df[C.ID_COL] == sid]
        print(f"  {sid}: {s[C.TIME_COL].min()} to {s[C.TIME_COL].max()}  ({len(s):,} rows)")

    # missing values
    print("\nMissing value rates:")
    for col in [C.TARGET] + C.AUX_COLS + C.COV_NUM_COLS + [C.COV_CAT_COL]:
        rate = df[col].isnull().mean()
        if rate > 0:
            print(f"  {col}: {rate:.1%}")
    if df[[C.TARGET] + C.AUX_COLS + C.COV_NUM_COLS].isnull().sum().sum() == 0:
        print("  (no missing values found)")

    # cov_cat categories
    cats = df[C.COV_CAT_COL].dropna().unique()
    print(f"\ncov_cat categories ({len(cats)}): {sorted(cats)}")

    # first few rows
    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    return True


# ──────────────────────────────────────────────
# STEP 3: Prepare the data (clean + split)
# ──────────────────────────────────────────────
def prepare_data(df):
    """Clean data and split into train/val/test by time."""
    print("\n" + "=" * 50)
    print("PREPARING DATA")
    print("=" * 50)

    # parse timestamps and sort
    df[C.TIME_COL] = pd.to_datetime(df[C.TIME_COL])
    df = (df.sort_values([C.ID_COL, C.TIME_COL])
            .drop_duplicates(subset=[C.ID_COL, C.TIME_COL], keep="last")
            .reset_index(drop=True))

    # handle missing cov_cat - explicit bucket
    df[C.COV_CAT_COL] = df[C.COV_CAT_COL].astype("string").fillna("__missing__")
    df[C.COV_CAT_COL] = pd.Categorical(df[C.COV_CAT_COL])
    print(f"cov_cat: {len(df[C.COV_CAT_COL].cat.categories)} categories (missing -> '__missing__')")

    # time-ordered split per series (70% train / 15% val / 15% test)
    train_list, val_list, test_list = [], [], []
    for sid, g in df.groupby(C.ID_COL, sort=False):
        g = g.sort_values(C.TIME_COL)
        n = len(g)
        i_tr = int(n * C.TRAIN_FRAC)
        i_va = int(n * (C.TRAIN_FRAC + C.VAL_FRAC))
        train_list.append(g.iloc[:i_tr])
        val_list.append(g.iloc[i_tr:i_va])
        test_list.append(g.iloc[i_va:])

    train = pd.concat(train_list).reset_index(drop=True)
    val = pd.concat(val_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True)

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train):,} rows  ({C.TRAIN_FRAC:.0%})")
    print(f"  Val:   {len(val):,} rows  ({C.VAL_FRAC:.0%})")
    print(f"  Test:  {len(test):,} rows  ({1 - C.TRAIN_FRAC - C.VAL_FRAC:.0%})")
    print(f"\nTrain ends at:  {train[C.TIME_COL].max()}")
    print(f"Val ends at:    {val[C.TIME_COL].max()}")
    print(f"Test ends at:   {test[C.TIME_COL].max()}")

    return train, val, test


# ──────────────────────────────────────────────
# OPTIONAL: Generate sample data for testing
# ──────────────────────────────────────────────
def generate_sample(n_series=10, n_hours=24 * 120):
    """Create sample CSVs matching the assignment schema."""
    rng = np.random.default_rng(C.SEED)
    start = pd.Timestamp("2013-03-01")
    cats = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

    for s in range(n_series):
        sid = f"series_{chr(65 + s)}"
        ts = pd.date_range(start, periods=n_hours, freq="h")
        t = np.arange(n_hours)
        cov1 = rng.normal(0, 1, n_hours).cumsum() / 10
        target = (50 + 10 * np.sin(2 * np.pi * t / 24)
                  + 5 * np.sin(2 * np.pi * t / 168)
                  + 0.01 * t + 3 * cov1
                  + rng.normal(0, 2, n_hours))
        row = {C.TIME_COL: ts, C.ID_COL: sid, C.TARGET: target}
        for i, c in enumerate(C.COV_NUM_COLS):
            row[c] = cov1 if i == 0 else rng.normal(0, 1, n_hours)
        for c in C.AUX_COLS:
            row[c] = rng.normal(0, 1, n_hours)
        row[C.COV_CAT_COL] = rng.choice(cats, n_hours)
        df = pd.DataFrame(row)
        # inject ~3% missing values
        for col in [C.TARGET] + C.AUX_COLS + C.COV_NUM_COLS:
            df.loc[rng.random(n_hours) < 0.03, col] = np.nan
        df.to_csv(C.DATA_RAW / f"{sid}.csv", index=False)

    print(f"Wrote {n_series} sample series to {C.DATA_RAW}")


# ──────────────────────────────────────────────
# Main: run all three steps in order
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true",
                        help="Generate sample data for testing")
    args = parser.parse_args()

    if args.sample:
        generate_sample()
        print()  # blank line before continuing

    # Step 1: Read
    df = read_data()
    if df is None:
        return

    # Step 2: Check
    ok = check_data(df)
    if not ok:
        return

    # Step 3: Prepare
    train, val, test = prepare_data(df)
    
    # Step 4: Save splits
    print("\n" + "=" * 50)
    print("SAVING PREPARED DATA")
    print("=" * 50)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(C.DATA_PROCESSED / "train.csv", index=False)
    val.to_csv(C.DATA_PROCESSED / "val.csv", index=False)
    test.to_csv(C.DATA_PROCESSED / "test.csv", index=False)
    
    print(f"Train saved: {C.DATA_PROCESSED / 'train.csv'}")
    print(f"Val saved:   {C.DATA_PROCESSED / 'val.csv'}")
    print(f"Test saved:  {C.DATA_PROCESSED / 'test.csv'}")
    print("\nDone. Data is ready for training.")


if __name__ == "__main__":
    main()