#!/usr/bin/env python3
"""
analyze_cv_penetration.py — CV Penetration Rate Analysis
Computes  penetration_rate = segment_journeys / ATSPM_total_volume
  by  signal-id  ×  time-of-day (15-min bins)

Outputs:
  1) CSV : cv_penetration_by_signal_tod.csv
  2) Heatmap : cv_penetration_heatmap.png
  3) Summary stats printed to stdout
"""

import os, re, ast, glob, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────
MATCHED_DIR  = '/data2/NPTOI/arterial/matched_metrics'
MAPPING_FILE = '/home/ya195253/YJ/python_scripts/ATSPM/data/ATSPM_CV_matched.csv'
OUTPUT_DIR   = '/home/ya195253/YJ/python_scripts/ATSPM/plot'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────
# 1. Load Seg → Signal mapping
# ──────────────────────────────────────────────────
def load_seg_to_signal():
    """Return dict: seg_unique(str) → signalID(str)"""
    mapping = pd.read_csv(MAPPING_FILE)
    seg_to_signal = {}
    for _, row in mapping.iterrows():
        seg = str(int(row['Seg_Unique']))
        sig = str(row['signalID'])
        seg_to_signal[seg] = sig
    return seg_to_signal, mapping

PHASE_RE = re.compile(r'ATSPM_volumePhase(\d+)')

def get_atspm_total_volume(df):
    """
    Sum all ATSPM_volumePhaseN columns (total phase volume, no suffix).
    Use bare columns first; fall back to _x / _y if bare are all NaN.
    """
    # Bare columns like ATSPM_volumePhase2, ATSPM_volumePhase6
    bare = [c for c in df.columns
            if PHASE_RE.match(c) and not c.endswith(('_x','_y'))
            and 'T' not in c.split('Phase')[1] and 'L' not in c.split('Phase')[1]
            and 'R' not in c.split('Phase')[1]]
    if bare:
        vol = df[bare].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
        if vol.notna().sum() > 0:
            return vol

    # Fallback: _x suffix
    x_cols = [c for c in df.columns
              if PHASE_RE.match(c.replace('_x','')) and c.endswith('_x')
              and 'T' not in c.split('Phase')[1].replace('_x','')
              and 'L' not in c.split('Phase')[1].replace('_x','')
              and 'R' not in c.split('Phase')[1].replace('_x','')]
    if x_cols:
        vol = df[x_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=1)
        if vol.notna().sum() > 0:
            return vol

    return pd.Series(np.nan, index=df.index)


# ──────────────────────────────────────────────────
# 2. Main collection loop
# ──────────────────────────────────────────────────
def collect_data():
    seg_to_signal, mapping_df = load_seg_to_signal()
    print(f"Mapping: {len(seg_to_signal)} segments → {len(set(seg_to_signal.values()))} signals")

    all_files = sorted(glob.glob(os.path.join(MATCHED_DIR, '*', '*.parquet')))
    print(f"Total files: {len(all_files)}")

    records = []

    for fi, fpath in enumerate(all_files):
        if fi % 100 == 0:
            print(f"  [{fi}/{len(all_files)}]...")
        try:
            pf = pd.read_parquet(fpath)
            keep = [c for c in pf.columns
                    if c in ('Seg_Unique','Time','segment_journeys','total_journeys')
                    or c.startswith('ATSPM_volume')]
            df = pf[keep]
            del pf
            if len(df) == 0:
                continue

            df['_seg'] = df['Seg_Unique'].apply(lambda x: str(int(x)) if pd.notna(x) else None)
            df = df.dropna(subset=['_seg','Time'])

            # Compute ATSPM total volume per row
            df['atspm_vol'] = get_atspm_total_volume(df)
            df['cv_journeys'] = pd.to_numeric(df['segment_journeys'], errors='coerce')

            for seg_id, seg_df in df.groupby('_seg'):
                signal_id = seg_to_signal.get(seg_id)
                if signal_id is None:
                    continue  # no mapping, skip

                sub = seg_df[['Time','cv_journeys','atspm_vol']].copy()
                sub['signalID'] = signal_id
                sub['seg'] = seg_id
                records.append(sub)

        except Exception as e:
            continue

    result = pd.concat(records, ignore_index=True)
    result['Time'] = pd.to_datetime(result['Time'])
    result['time_of_day'] = result['Time'].dt.strftime('%H:%M')
    result['date'] = result['Time'].dt.date
    result['hour'] = result['Time'].dt.hour

    # Penetration rate
    result['penetration_rate'] = result['cv_journeys'] / result['atspm_vol']
    # Cap at 1.0 (detector undercounts can cause > 1)
    result['penetration_rate'] = result['penetration_rate'].clip(upper=1.0)

    print(f"\nTotal rows collected: {len(result):,}")
    return result


# ──────────────────────────────────────────────────
# 3. Analysis & Visualization
# ──────────────────────────────────────────────────
def analyze(df):
    print("\n" + "="*70)
    print("CV PENETRATION RATE ANALYSIS")
    print("="*70)

    valid = df.dropna(subset=['penetration_rate'])
    valid = valid[valid['penetration_rate'] > 0]
    print(f"\nValid rows (PR > 0): {len(valid):,} / {len(df):,}")
    print(f"Unique signals: {valid['signalID'].nunique()}")
    print(f"Date range: {valid['date'].min()} ~ {valid['date'].max()}")

    # ── A. Overall statistics ──
    print(f"\n── Overall Penetration Rate ──")
    print(f"  Mean:   {valid['penetration_rate'].mean():.4f} ({valid['penetration_rate'].mean()*100:.2f}%)")
    print(f"  Median: {valid['penetration_rate'].median():.4f} ({valid['penetration_rate'].median()*100:.2f}%)")
    print(f"  Std:    {valid['penetration_rate'].std():.4f}")
    print(f"  Q25:    {valid['penetration_rate'].quantile(0.25):.4f}")
    print(f"  Q75:    {valid['penetration_rate'].quantile(0.75):.4f}")

    # ── B. By Signal ID ──
    by_signal = valid.groupby('signalID')['penetration_rate'].agg(['mean','median','std','count'])
    by_signal.columns = ['mean_PR','median_PR','std_PR','n_obs']
    by_signal = by_signal.sort_values('mean_PR', ascending=False)
    print(f"\n── Top 20 Signals (by mean PR) ──")
    print(by_signal.head(20).to_string())
    print(f"\n── Bottom 10 Signals (by mean PR) ──")
    print(by_signal.tail(10).to_string())

    # ── C. By Time-of-Day ──
    by_tod = valid.groupby('time_of_day')['penetration_rate'].agg(['mean','median','std','count'])
    by_tod.columns = ['mean_PR','median_PR','std_PR','n_obs']
    print(f"\n── By Time-of-Day (sample) ──")
    print(by_tod.to_string())

    # ── D. Heatmap: Signal × ToD ──
    pivot = valid.groupby(['signalID','time_of_day'])['penetration_rate'].mean().unstack(fill_value=np.nan)

    # Sort signal IDs by overall mean PR
    sig_order = by_signal.index.tolist()
    pivot = pivot.reindex(sig_order)

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, 'cv_penetration_by_signal_tod.csv')
    pivot.to_csv(csv_path)
    print(f"\nSaved pivot CSV → {csv_path}")

    # ── E. Plot: Time-of-Day profile ──
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), dpi=150)

    # E1: Average PR by time of day
    ax = axes[0]
    tod_sorted = by_tod.sort_index()
    x = range(len(tod_sorted))
    ax.plot(x, tod_sorted['mean_PR']*100, color='#2196F3', linewidth=2, label='Mean')
    ax.fill_between(x,
                    (tod_sorted['mean_PR']-tod_sorted['std_PR']).clip(0)*100,
                    (tod_sorted['mean_PR']+tod_sorted['std_PR']).clip(upper=1)*100,
                    alpha=0.15, color='#2196F3')
    ax.plot(x, tod_sorted['median_PR']*100, color='#FF5722', linewidth=1.5, linestyle='--', label='Median')
    ax.set_xticks(x[::4])
    ax.set_xticklabels(tod_sorted.index[::4], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Penetration Rate (%)', fontsize=12)
    ax.set_title('CV Penetration Rate by Time-of-Day (all signals)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # E2: Heatmap (top 40 signals)
    ax2 = axes[1]
    top_n = min(40, len(sig_order))
    hm_data = pivot.iloc[:top_n]
    im = ax2.imshow(hm_data.values * 100, aspect='auto', cmap='YlOrRd',
                    interpolation='nearest')
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(hm_data.index, fontsize=6)
    tod_labels = hm_data.columns
    ax2.set_xticks(range(0, len(tod_labels), 4))
    ax2.set_xticklabels(tod_labels[::4], rotation=45, ha='right', fontsize=7)
    ax2.set_xlabel('Time-of-Day', fontsize=11)
    ax2.set_ylabel('Signal ID', fontsize=11)
    ax2.set_title(f'CV Penetration Rate Heatmap (Top {top_n} Signals)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Penetration Rate (%)', fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'cv_penetration_heatmap.png')
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"Saved figure → {fig_path}")

    # ── F. Distribution by hour ──
    fig2, ax3 = plt.subplots(figsize=(14, 6), dpi=150)
    valid.boxplot(column='penetration_rate', by='hour', ax=ax3,
                  showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor='#42A5F5', alpha=0.7),
                  medianprops=dict(color='#D32F2F', linewidth=2))
    ax3.set_xlabel('Hour of Day', fontsize=12)
    ax3.set_ylabel('Penetration Rate', fontsize=12)
    ax3.set_title('CV Penetration Rate Distribution by Hour', fontsize=14, fontweight='bold')
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.suptitle('')
    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, 'cv_penetration_boxplot_hour.png')
    plt.savefig(fig2_path, bbox_inches='tight')
    plt.close()
    print(f"Saved figure → {fig2_path}")

    # ── G. By Road (corridor) ──
    mapping = pd.read_csv(MAPPING_FILE)
    mapping['_seg'] = mapping['Seg_Unique'].apply(lambda x: str(int(x)))
    seg_road = mapping.set_index('_seg')['Seg_Road_name'].to_dict()
    valid['road'] = valid['seg'].map(seg_road)
    by_road = valid.groupby('road')['penetration_rate'].agg(['mean','median','count'])
    by_road.columns = ['mean_PR','median_PR','n_obs']
    by_road = by_road.sort_values('mean_PR', ascending=False)
    print(f"\n── By Road / Corridor ──")
    print(by_road.to_string())

    return by_signal, by_tod


# ──────────────────────────────────────────────────
if __name__ == '__main__':
    df = collect_data()
    by_signal, by_tod = analyze(df)
    print("\n✅ Analysis complete!")
