import os
import pandas as pd
import glob
import ast
import numpy as np

base_dir = '/home/ya195253/YJ/python_scripts/ATSPM'
signal_dir = '/data2/ATSPM/config_data/signal_config_2025'
mapping_path = os.path.join(base_dir, 'data', 'ATSPM_CV_matched.csv')

print("Loading mapping...")
match_df = pd.read_csv(mapping_path)
match_df['signalID'] = match_df['signalID'].astype(str)

results = []

def parse_phase(s):
    try:
        return ast.literal_eval(s)
    except:
        return []

for idx, row in match_df.iterrows():
    sig = str(row['signalID'])
    seg = str(row['Seg_Unique'])
    
    thru = set(parse_phase(row['Through']) + parse_phase(row.get('Through Right', '[]')))
    left = set(parse_phase(row['Left']) + parse_phase(row.get('Left Through', '[]')))
    all_phases = thru.union(left)
    
    # Load lanes
    lane_f = os.path.join(signal_dir, 'lanes', f"{sig}.csv")
    app_f = os.path.join(signal_dir, 'approach', f"{sig}.csv")
    
    thru_lanes = 0
    left_lanes = 0
    speed_limit = 45 # default
    
    if os.path.exists(lane_f) and len(all_phases) > 0:
        try:
            lanes = pd.read_csv(lane_f)
            lanes.columns = lanes.columns.str.strip()
            
            if 'Phase' in lanes.columns and 'Lane Type' in lanes.columns:
                lanes['Phase'] = pd.to_numeric(lanes['Phase'], errors='coerce')
                lanes = lanes[lanes['Phase'].notna()]
                
                # Count lanes matched to the exact phases
                thru_lanes = len(lanes[lanes['Phase'].isin(thru) & lanes['Lane Type'].str.contains('Through', na=False, case=False)])
                left_lanes = len(lanes[lanes['Phase'].isin(left) & lanes['Lane Type'].str.contains('Left', na=False, case=False)])
                
                # Get approach speed limit
                if os.path.exists(app_f):
                    app = pd.read_csv(app_f)
                    app.columns = app.columns.str.strip()
                    # Find which approach(es) the phases belong to in the lane df
                    if 'Approach' in lanes.columns:
                        matched_app = lanes[lanes['Phase'].isin(all_phases)]['Approach'].unique()
                        if len(matched_app) > 0 and 'Speed Limit' in app.columns and 'Approach' in app.columns:
                            # Approach string example: '1 I-4 WB Ramp NEB'
                            # Find matching row in app_df
                            for a in matched_app:
                                a_prefix = str(a).split(' ')[0] # "1"
                                match_app = app[app['Approach'].astype(str).str.startswith(a_prefix + " ")]
                                if len(match_app) > 0:
                                    try:
                                        sl = float(match_app['Speed Limit'].iloc[0])
                                        if not np.isnan(sl):
                                            speed_limit = sl
                                    except:
                                        pass
        except Exception as e:
            print(f"Error processing {sig}: {e}")

    results.append({
        'Seg_Unique': seg,
        'signalID': sig,
        'thru_lanes': thru_lanes,
        'left_lanes': left_lanes,
        'speed_limit_mph': speed_limit
    })

res_df = pd.DataFrame(results)
print(f"Extracted {len(res_df)} features. Examples:")
print(res_df.head(10))

# Save to stgnn network folder
out_path = '/data2/NPTOI/arterial/imputation_network/physical_features.csv'
res_df.to_csv(out_path, index=False)
print(f"Saved to {out_path}")
