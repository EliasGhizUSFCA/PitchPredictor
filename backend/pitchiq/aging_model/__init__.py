"""System B: pitch characteristic & aging forecast engine (backtested population + pitcher-level forecasts).

See docs/aging-model.md. Reads the aging dataset (pitchiq.aging, tables pitcher_season_stats /
pitcher_season_pitch_stats / pitcher_bio) and the target pitcher's pitch-level rows; writes its own tables
(pitchiq.models_aging_forecast)."""
