"""Tests for the data-preparation helpers in pipeline.db.seed_reference (no database)."""

import pandas as pd

from pipeline.db.seed_reference import (
    attach_district_id,
    build_feature_rows,
    long_frame,
    slugify,
)

IDS = {("Punjab", "Rupnagar"): 7, ("Punjab", "Nawanshahr"): 8, ("Haryana", "Ambala"): 1}


def test_slugify():
    assert slugify("Uttar Pradesh") == "uttar-pradesh"
    assert slugify("Tarn Taran") == "tarn-taran"


def test_aliases_map_to_district_list_spelling():
    df = pd.DataFrame(
        {
            "state": ["Punjab", "Punjab"],
            "district": ["Ropar", "Shahid Bhagat Singh Nagar"],
        }
    )
    out = attach_district_id(df, IDS, "test")
    assert out["district_id"].tolist() == [7, 8]


def test_unknown_district_is_skipped_not_fatal():
    df = pd.DataFrame({"state": ["Punjab", "Haryana"], "district": ["Mohali", "Ambala"]})
    out = attach_district_id(df, IDS, "test")
    assert out["district_id"].tolist() == [1]


def test_long_frame_drops_missing_values():
    df = pd.DataFrame({"district_id": [1, 2], "year": [2019, 2019], "ndvi_kharif": [0.5, None]})
    out = long_frame(df, [("ndvi_kharif", "Kharif", "ndvi")])
    assert len(out) == 1
    assert out.iloc[0]["metric"] == "ndvi"
    assert out.iloc[0]["season"] == "Kharif"


def test_feature_json_holds_null_not_nan():
    model_df = pd.DataFrame(
        {
            "state": ["Haryana"],
            "district": ["Ambala"],
            "crop": ["Wheat"],
            "season": ["Rabi"],
            "year": [2019],
            "yield_kg_ha": [4000.0],
            "area_ha": [1000.0],
            "lag_yield_1": [float("nan")],
            "time_trend": [22.0],
        }
    )
    rows = build_feature_rows(model_df, IDS, ["lag_yield_1", "time_trend"])
    assert len(rows) == 1
    assert rows[0][6].obj == {"lag_yield_1": None, "time_trend": 22.0}
