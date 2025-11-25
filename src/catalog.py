import pandas as pd
import numpy as np

def load_catalog(path, mag_limit=None):
    
    df = pd.read_csv(path)

    # If compact format with unit vectors exists, use it and avoid recomputing
    if all(c in df.columns for c in ("ux", "uy", "uz")):
        df = df.reset_index(drop=True)
        df['x'] = df['ux'].astype(float)
        df['y'] = df['uy'].astype(float)
        df['z'] = df['uz'].astype(float)
    else:
        # Determine RA/Dec source
        if 'ra_deg' in df.columns and 'dec_deg' in df.columns:
            ra_deg = pd.to_numeric(df['ra_deg'], errors='coerce')
            dec_deg = pd.to_numeric(df['dec_deg'], errors='coerce')
            ra = np.deg2rad(ra_deg.fillna(0.0).astype(float))
            dec = np.deg2rad(dec_deg.fillna(0.0).astype(float))
        else:
            # Fall back to 'ra'/'dec' columns; detect hours vs degrees for RA
            ra_raw = pd.to_numeric(df.get('ra', pd.Series(np.nan)), errors='coerce')
            dec_raw = pd.to_numeric(df.get('dec', pd.Series(np.nan)), errors='coerce')
            if ra_raw.max(skipna=True) <= 24.1:
                ra = np.deg2rad(ra_raw.fillna(0.0).astype(float) * 15.0)
            else:
                ra = np.deg2rad(ra_raw.fillna(0.0).astype(float))
            dec = np.deg2rad(dec_raw.fillna(0.0).astype(float))

        x = np.cos(dec) * np.cos(ra)
        y = np.cos(dec) * np.sin(ra)
        z = np.sin(dec)
        df['x'] = x
        df['y'] = y
        df['z'] = z

    # Filter by magnitude
    if mag_limit is not None and 'mag' in df.columns:
        df = df[pd.to_numeric(df['mag'], errors='coerce') <= mag_limit]

    out = df[['mag', 'x', 'y', 'z']].copy()
    out = out.reset_index(drop=True)
    return out
