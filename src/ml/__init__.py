# src/ml: global LightGBM forecasting.
#
# Layout:
#   dataset.py   load, split, and select stratified validation SKUs
#   model.py     feature matrix + LGBMForecaster (fit/predict)
#   evaluate.py  pooled-WAPE scoring and bootstrap significance, model-agnostic
#
# Experiments live in scripts/ml_*.py and only compose these pieces.
