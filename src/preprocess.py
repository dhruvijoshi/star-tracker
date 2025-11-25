#!/usr/bin/env python3
"""Preprocess the HYG star catalogue CSV.

Creates a cleaned CSV with numeric types, computed RA/Dec degrees
from radian columns when available, optional magnitude filtering,
and basic missing-value handling. Also supports a compact mode that
computes unit vectors and optionally builds/saves a KD-tree.
"""
import argparse
import math
import sys

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame, mag_cut: float | None = None, drop_high_dist=True) -> pd.DataFrame:
    # Convert obvious numeric columns
    numeric_cols = [
        "ra",
        "dec",
        "dist",
        "pmra",
        "pmdec",
        "rv",
        "mag",
        "absmag",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "rarad",
        "decrad",
        "pmrarad",
        "pmdecrad",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Some catalogue entries use sentinel large values (e.g. 100000.0) for unknown distances
    if "dist" in df.columns and drop_high_dist:
        df.loc[df["dist"] >= 1e5, "dist"] = np.nan

    # Remove exact duplicates by 'id' if present, otherwise by (hip,hd,hr)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"]).copy()
    else:
        subset = [c for c in ("hip", "hd", "hr") if c in df.columns]
        if subset:
            df = df.drop_duplicates(subset=subset).copy()

    # Compute RA/Dec in degrees from radian columns if available
    if "rarad" in df.columns and "decrad" in df.columns:
        df["ra_deg"] = np.degrees(df["rarad"])
        df["dec_deg"] = np.degrees(df["decrad"])
    else:
        df["ra_deg"] = np.nan
        df["dec_deg"] = np.nan

    # If ra_deg missing but 'ra' exists and looks like hours (values < 24), convert hours->deg
    if "ra" in df.columns:
        ra_numeric = pd.to_numeric(df["ra"], errors="coerce")
        mask = df["ra_deg"].isna() & ra_numeric.notna() & (ra_numeric.abs() <= 24)
        df.loc[mask, "ra_deg"] = ra_numeric[mask] * 15.0

    # If dec_deg missing but 'dec' exists assume degrees
    if "dec" in df.columns:
        dec_numeric = pd.to_numeric(df["dec"], errors="coerce")
        mask2 = df["dec_deg"].isna() & dec_numeric.notna() & (dec_numeric.abs() <= 90)
        df.loc[mask2, "dec_deg"] = dec_numeric[mask2]

    # Basic filters
    if mag_cut is not None and "mag" in df.columns:
        df = df[df["mag"] <= mag_cut].copy()

    # Reorder: put id/identifiers and coordinate columns first
    cols = list(df.columns)
    pref = [c for c in ("id", "hip", "hd", "proper") if c in cols]
    coord = [c for c in ("ra_deg", "dec_deg", "rarad", "decrad", "ra", "dec") if c in cols]
    rest = [c for c in cols if c not in pref + coord]
    new_order = pref + coord + rest
    df = df.reindex(columns=new_order)

    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess HYG catalogue CSV")
    parser.add_argument("input", nargs="?", default="data/catalog/hygdata_v42.csv", help="Input CSV path")
    parser.add_argument("--output", "-o", default="data/catalog/hygdata_v42_clean.csv", help="Output CSV path")
    parser.add_argument("--mag-cut", type=float, default=None, help="Optional: keep stars with mag <= MAG_CUT")
    parser.add_argument("--no-drop-high-dist", dest="drop_high_dist", action="store_false", help="Do not treat large sentinel distances as missing")
    # Compact/export options
    parser.add_argument("--compact", action="store_true", help="Produce a compact filtered catalog with unit vectors")
    parser.add_argument("--dist-cut", type=float, default=None, help="Optional: keep stars with dist <= DIST_CUT (pc)")
    parser.add_argument("--save-npz", dest="save_npz", default=None, help="Save compact arrays to .npz file")
    parser.add_argument("--save-tree", default=None, help="Save KD-tree (pickle) to this path (requires scipy)")

    args = parser.parse_args(argv)

    print(f"Loading '{args.input}'...")
    df = pd.read_csv(args.input, low_memory=False)
    print("Shape before:", df.shape)

    cleaned = preprocess(df, mag_cut=args.mag_cut, drop_high_dist=args.drop_high_dist)

    print("Shape after:", cleaned.shape)
    # Print quick diagnostics
    important = ["mag", "ra_deg", "dec_deg", "dist"]
    for c in important:
        if c in cleaned.columns:
            nmiss = int(cleaned[c].isna().sum())
            print(f"{c}: missing {nmiss} / {len(cleaned)}")

    cleaned.to_csv(args.output, index=False)
    print(f"Wrote cleaned CSV to '{args.output}'")

    if args.compact:
        # Defaults: if user didn't pass cuts, use conservative visible-star defaults
        mag_cut = args.mag_cut if args.mag_cut is not None else 6.0
        dist_cut = args.dist_cut if args.dist_cut is not None else 200.0

        # Create mask: require ra_deg, dec_deg, mag present
        mask = cleaned["ra_deg"].notna() & cleaned["dec_deg"].notna()
        if "mag" in cleaned.columns:
            mask &= cleaned["mag"].notna() & (cleaned["mag"] <= mag_cut)
        if "dist" in cleaned.columns:
            # keep rows where dist is NaN (unknown) OR dist <= dist_cut
            mask &= cleaned["dist"].isna() | (cleaned["dist"] <= dist_cut)

        compact = cleaned[mask].copy()
        print(f"Compact selection: mag<={mag_cut}, dist<={dist_cut} -> {len(compact)} rows")

        # Compute unit vectors
        ra_rad = np.radians(compact["ra_deg"].to_numpy(dtype=float))
        dec_rad = np.radians(compact["dec_deg"].to_numpy(dtype=float))
        cosdec = np.cos(dec_rad)
        ux = cosdec * np.cos(ra_rad)
        uy = cosdec * np.sin(ra_rad)
        uz = np.sin(dec_rad)
        compact["ux"] = ux
        compact["uy"] = uy
        compact["uz"] = uz

        # Select minimal columns
        prefer = [c for c in ("id", "hip", "hd", "proper") if c in compact.columns]
        cols_out = prefer + ["ra_deg", "dec_deg", "mag", "dist", "ux", "uy", "uz"]
        cols_out = [c for c in cols_out if c in compact.columns]
        compact_out = compact[cols_out]

        out_csv = args.output.replace("_clean.csv", "_compact.csv") if args.output.endswith("_clean.csv") else args.output + ".compact.csv"
        compact_out.to_csv(out_csv, index=False)
        print(f"Wrote compact CSV to '{out_csv}'")

        # Save arrays to npz if requested
        if args.save_npz:
            np.savez_compressed(args.save_npz, id=compact_out.get("id"), hip=compact_out.get("hip"), hd=compact_out.get("hd"), ra_deg=compact_out["ra_deg"].to_numpy(), dec_deg=compact_out["dec_deg"].to_numpy(), mag=compact_out.get("mag"), dist=compact_out.get("dist"), ux=compact_out["ux"].to_numpy(), uy=compact_out["uy"].to_numpy(), uz=compact_out["uz"].to_numpy())
            print(f"Saved compact arrays to '{args.save_npz}'")

        # Build KD-tree if requested (requires scipy)
        if args.save_tree:
            try:
                from scipy.spatial import cKDTree
                import pickle

                tree = cKDTree(np.vstack((ux, uy, uz)).T)
                with open(args.save_tree, "wb") as fh:
                    pickle.dump({"tree": tree, "ids": compact_out.get("id"), "hip": compact_out.get("hip"), "hd": compact_out.get("hd")}, fh, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved KD-tree pickle to '{args.save_tree}'")
            except Exception as e:
                print("Could not build/save KD-tree (scipy may be missing). Saved arrays only.", e)


if __name__ == "__main__":
    main()
