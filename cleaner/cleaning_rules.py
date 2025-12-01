# cleaning_rules.py
import pandas as pd
import numpy as np

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # ---- select the ML columns you want to keep ----
    keep = [
        "crash_record_id","crash_type","lighting_condition",
        "posted_speed_limit","road_defect","roadway_surface_cond","trafficway_type",
        "veh_count","ppl_count","injuries_total","crash_date","weather_condition",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # your existing mappings, just applied to df instead of reading from disk:
    mapping_1 = {'injury and / or tow due to crash': 1, 'no injury / drive away': 0}
    if "crash_type" in df.columns:
        df["crash_type"] = (
            df["crash_type"].astype(str).str.strip().str.lower()
            .map(mapping_1).astype("Int64")
        )

    # roadway_surface_cond
    if "roadway_surface_cond" in df.columns:
        col = (df["roadway_surface_cond"].astype("string").str.strip().str.lower()
               .str.replace(r"\s*,\s*", ", ", regex=True).str.replace(r"\s+", " ", regex=True))
        mapping_2 = {'dry':'dry','wet':'wet','sand, mud, dirt':'loose','unknown':pd.NA,'other':'other'}
        df["roadway_surface_cond"] = col.map(mapping_2)

    # lighting_condition
    if "lighting_condition" in df.columns:
        col = (df["lighting_condition"].astype("string").str.strip().str.lower()
               .str.replace(r"\s*,\s*", ", ", regex=True).str.replace(r"\s+", " ", regex=True))
        mapping_3 = {'daylight':'daylight','dawn':'dawn','dusk':'dusk',
                     'darkness, lighted road':'dark_lighted','darkness':'dark_unlighted','unknown':pd.NA}
        df["lighting_condition"] = col.map(mapping_3)

    # road_defect
    if "road_defect" in df.columns:
        col = (df["road_defect"].astype("string").str.strip().str.lower()
               .str.replace(r"\s*,\s*", ", ", regex=True).str.replace(r"\s+", " ", regex=True))
        mapping_4 = {'no defects':'none','unknown':pd.NA,'other':'other',
                     'rut, holes':'potholes_ruts','shoulder defect':'shoulder','worn surface':'surface_wear',
                     'debris on roadway':'debris'}
        df["road_defect"] = col.map(mapping_4)

    # trafficway_type
    if "trafficway_type" in df.columns:
        col = (df["trafficway_type"].astype("string").str.strip().str.lower()
               .str.replace(r"\s*,\s*", ", ", regex=True).str.replace(r"\s*-\s*", " - ", regex=True)
               .str.replace(r"\s+", " ", regex=True))
        mapping_5 = {'not divided':'undivided','divided - w/median barrier':'divided_barrier',
                     'divided - w/median (not raised)':'divided_median','one-way':'one_way',
                     't-intersection':'intersection_t','four way':'intersection_4_way',
                     'five point, or more':'intersection_5_plus','unknown intersection type':pd.NA,
                     'roundabout':'roundabout','parking lot':'parking_lot','alley':'alley',
                     'traffic route':'traffic_route','center turn lane':'center_turn_lane','ramp':'ramp',
                     'other':'other','unknown':pd.NA}
        df["trafficway_type"] = col.map(mapping_5)

    # parse timestamp if present
    if "crash_date" in df.columns:
        df["crash_date"] = pd.to_datetime(df["crash_date"], errors="coerce", utc=True)

    # normalize column names for downstream
    df.columns = (df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True))

    # basic null discipline on key
    df = df[df["crash_record_id"].notna()]

    return df
