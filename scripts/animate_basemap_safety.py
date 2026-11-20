#!/usr/bin/env python3
"""
animate_basemap_safety.py — Animate hourly Dilemma Zone & Red Light Running
ratios on the road network basemap.

Pipeline:
  1. Load all parquet files from matched_metrics
  2. Aggregate to 1-hour bins per segment, then average across ALL days
  3. Join with basemap shapefile geometry
  4. Render 24 frames (0:00–23:00) as a ~10s MP4 video
"""

import os, glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.lines import Line2D
import contextily as cx
import warnings
warnings.filterwarnings('ignore')

# ── paths ────────────────────────────────────────────────────────────────────
MATCHED_DIR = '/data2/NPTOI/arterial/matched_metrics'
BASEMAP_SHP = '/data2/NPTOI/arterial/basemap/OS_Primary_Basemap_segments.shp'
OUTPUT_DIR  = '/data2/NPTOI/arterial/visualization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── target columns ───────────────────────────────────────────────────────────
# CV-based ratio columns (0-1 range, already ratios)
DILEMMA_COLS = ['through_dilemma', 'left_dilemma']
RLR_COLS     = ['through_RLR', 'left_RLR']


def load_all_data():
    """Load all parquet files and extract relevant columns."""
    files = sorted(glob.glob(os.path.join(MATCHED_DIR, '*', '*.parquet')))
    print(f"Loading {len(files)} files...")

    all_dfs = []
    for i, fpath in enumerate(files):
        if i % 100 == 0:
            print(f"  [{i}/{len(files)}]")
        try:
            cols_needed = ['Seg_Unique', 'Time'] + DILEMMA_COLS + RLR_COLS
            df = pd.read_parquet(fpath, columns=cols_needed)
            if len(df) == 0:
                continue
            all_dfs.append(df)
        except Exception as e:
            continue

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all['Time'] = pd.to_datetime(df_all['Time'])
    print(f"  Total rows: {len(df_all):,}")
    return df_all


def hourly_avg_across_days(df_all):
    """
    1. Floor each timestamp to hour-of-day (0-23).
    2. For each segment × hour, take the mean across all days.
    Returns: DataFrame with (Seg_Unique, hour, dilemma_ratio, rlr_ratio).
    """
    df = df_all.copy()

    # Combine through + left: take max of non-NaN values per row
    df['dilemma_ratio'] = df[DILEMMA_COLS].max(axis=1)
    df['rlr_ratio'] = df[RLR_COLS].max(axis=1)

    df['hour'] = df['Time'].dt.hour

    # Group by segment × hour, average across all 15-min slots & all days
    agg = (
        df.groupby(['Seg_Unique', 'hour'])[['dilemma_ratio', 'rlr_ratio']]
        .mean()
        .reset_index()
    )
    print(f"  Hourly-avg shape: {agg.shape}")
    print(f"  Dilemma ratio range: [{agg['dilemma_ratio'].min():.4f}, {agg['dilemma_ratio'].max():.4f}]")
    print(f"  RLR ratio range:    [{agg['rlr_ratio'].min():.6f}, {agg['rlr_ratio'].max():.6f}]")
    return agg


def build_animation(agg, gdf):
    """Build a side-by-side animated map: Dilemma Zone | Red Light Running."""

    # Merge geometry
    gdf = gdf[['Seg_Unique', 'geometry']].copy()
    gdf['Seg_Unique'] = gdf['Seg_Unique'].astype(str)
    agg['Seg_Unique'] = agg['Seg_Unique'].astype(str)

    # Project to Web Mercator for contextily basemap
    gdf_wm = gdf.to_crs(epsg=3857)

    # Pre-merge for all hours
    merged = gdf_wm.merge(agg, on='Seg_Unique', how='inner')
    print(f"  Merged segments: {merged['Seg_Unique'].nunique()}")

    # ── color scales ─────────────────────────────────────────────────────
    # Dilemma: 0 → green, high → orange/red
    dilemma_cmap = mcolors.LinearSegmentedColormap.from_list(
        'dilemma', ['#2ecc71', '#f1c40f', '#e74c3c'], N=256
    )
    # RLR: 0 → blue, high → red
    rlr_cmap = mcolors.LinearSegmentedColormap.from_list(
        'rlr', ['#3498db', '#e67e22', '#c0392b'], N=256
    )

    # Global max for consistent scale
    vmax_d = max(agg['dilemma_ratio'].quantile(0.98), 0.01)
    vmax_r = max(agg['rlr_ratio'].quantile(0.98), 0.001)

    # ── figure setup ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='#1a1a2e')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08, wspace=0.05)

    def setup_ax(ax, title):
        ax.set_facecolor('#16213e')
        ax.set_title(title, fontsize=16, fontweight='bold', color='white', pad=12)
        ax.set_axis_off()

    setup_ax(ax1, 'Dilemma Zone Ratio')
    setup_ax(ax2, 'Red Light Running Ratio')

    # Add basemap
    bounds = gdf_wm.total_bounds
    pad_x = (bounds[2] - bounds[0]) * 0.05
    pad_y = (bounds[3] - bounds[1]) * 0.05
    for ax in [ax1, ax2]:
        ax.set_xlim(bounds[0] - pad_x, bounds[2] + pad_x)
        ax.set_ylim(bounds[1] - pad_y, bounds[3] + pad_y)

    # Try to add tile basemap
    try:
        for ax in [ax1, ax2]:
            cx.add_basemap(ax, source=cx.providers.CartoDB.DarkMatter, zoom=11)
    except Exception as e:
        print(f"  ⚠ Basemap tiles failed ({e}), using plain background")
        # Draw all road segments in dim gray as fallback
        gdf_wm.plot(ax=ax1, color='#333', linewidth=0.3)
        gdf_wm.plot(ax=ax2, color='#333', linewidth=0.3)

    # ── colorbars ────────────────────────────────────────────────────────
    sm_d = plt.cm.ScalarMappable(cmap=dilemma_cmap, norm=plt.Normalize(0, vmax_d))
    sm_r = plt.cm.ScalarMappable(cmap=rlr_cmap, norm=plt.Normalize(0, vmax_r))
    sm_d.set_array([])
    sm_r.set_array([])

    cbar1 = fig.colorbar(sm_d, ax=ax1, shrink=0.6, aspect=20, pad=0.02)
    cbar2 = fig.colorbar(sm_r, ax=ax2, shrink=0.6, aspect=20, pad=0.02)
    for cb in [cbar1, cbar2]:
        cb.ax.yaxis.set_tick_params(color='white')
        plt.setp(cb.ax.yaxis.get_ticklabels(), color='white', fontsize=10)

    # Suptitle (updated per frame)
    suptitle = fig.suptitle('', fontsize=22, fontweight='bold', color='white', y=0.96)

    # Storage for plotted artists
    artists = {'d_lines': [], 'r_lines': []}

    def animate(hour):
        # Remove previous artists
        for a in artists['d_lines']:
            a.remove()
        for a in artists['r_lines']:
            a.remove()
        artists['d_lines'] = []
        artists['r_lines'] = []

        suptitle.set_text(f'Hourly Safety Metrics — {hour:02d}:00 – {hour:02d}:59')

        hour_data = merged[merged['hour'] == hour]

        if len(hour_data) > 0:
            # Dilemma Zone
            gdf_h = gpd.GeoDataFrame(hour_data, geometry='geometry', crs=gdf_wm.crs)

            # Plot segments with no data as thin gray lines
            no_data_d = gdf_h[gdf_h['dilemma_ratio'].isna()]
            no_data_r = gdf_h[gdf_h['rlr_ratio'].isna()]
            has_data_d = gdf_h[gdf_h['dilemma_ratio'].notna()]
            has_data_r = gdf_h[gdf_h['rlr_ratio'].notna()]

            if len(no_data_d) > 0:
                a = no_data_d.plot(ax=ax1, color='#444', linewidth=0.5)
            if len(no_data_r) > 0:
                a = no_data_r.plot(ax=ax2, color='#444', linewidth=0.5)

            if len(has_data_d) > 0:
                vals_d = has_data_d['dilemma_ratio'].clip(0, vmax_d)
                colors_d = dilemma_cmap(vals_d / vmax_d)
                widths_d = 1.5 + 3.5 * (vals_d / vmax_d)
                for idx_row, (_, row) in enumerate(has_data_d.iterrows()):
                    line, = ax1.plot(*row.geometry.xy, color=colors_d[idx_row],
                                     linewidth=widths_d.iloc[idx_row], solid_capstyle='round')
                    artists['d_lines'].append(line)

            if len(has_data_r) > 0:
                vals_r = has_data_r['rlr_ratio'].clip(0, vmax_r)
                colors_r = rlr_cmap(vals_r / vmax_r)
                widths_r = 1.5 + 3.5 * (vals_r / vmax_r)
                for idx_row, (_, row) in enumerate(has_data_r.iterrows()):
                    line, = ax2.plot(*row.geometry.xy, color=colors_r[idx_row],
                                     linewidth=widths_r.iloc[idx_row], solid_capstyle='round')
                    artists['r_lines'].append(line)

        return []

    hours = list(range(24))
    anim = FuncAnimation(fig, animate, frames=hours, interval=420, blit=False)

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'safety_basemap_animation.mp4')
    writer = FFMpegWriter(fps=2.4, bitrate=3000)
    anim.save(out_path, writer=writer, dpi=120)
    print(f"\n✅ Saved: {out_path}")
    plt.close(fig)
    return out_path


def main():
    print("=" * 60)
    print("  Basemap Safety Animation: Dilemma Zone & RLR")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/3] Loading matched metrics...")
    df_all = load_all_data()

    # Step 2: Hourly aggregation (avg across all days)
    print("\n[2/3] Hourly aggregation (all-day average)...")
    agg = hourly_avg_across_days(df_all)

    # Step 3: Load basemap and animate
    print("\n[3/3] Building animation...")
    gdf = gpd.read_file(BASEMAP_SHP)
    out_path = build_animation(agg, gdf)

    print(f"\n{'='*60}")
    print(f"  Video: {out_path}")
    print(f"  24 frames, ~10s @ 2.4 fps")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
