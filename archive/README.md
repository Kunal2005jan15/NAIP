# Archive

Retired scripts kept for history. Nothing in `pipeline/`, `api/`, `dashboard/` or `tests/`
imports from this folder, and it is not run by CI.

| Script | Why retired |
|---|---|
| 03_explore_data, 05_feature_engineering, 06_train_model | Pre-v2 exploration and training, superseded by 17/18/19/20 |
| 29_test_second_snapshot, 30_debug_watch_merge, debug_district_coords_merge, 33_verify_kharif_risk, diagnose_kharif, check_days | One-off debugging scripts |
| setup_tavily | Wrote a key file. The key now comes from the `TAVILY_API_KEY` environment variable |