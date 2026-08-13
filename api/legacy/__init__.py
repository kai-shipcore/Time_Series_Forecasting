"""The statsforecast API surface, kept as a record of the work rather than as live code.

This package is the API half of the original statistical forecasting track. Its
model half is `src/legacy/`. Together they are the prototype that the LightGBM
system in `src/ml/` was built to beat, and they are retained deliberately: the
model selection, the cross-validation backtest and the conformal intervals are a
substantial part of what this project did, and deleting them would leave the
repository describing an outcome with no evidence of how it was reached.

WHAT TO EXPECT WHEN READING IT
------------------------------
`routes.py` holds sixteen endpoints. They served two screens in the Demand Pilot
app: the Demand Forecast page, and the per-SKU chart on SKU Planning. Both were
retired in August 2026, so nothing calls them now.

The code is not maintained. It is not run in production. It is not the
recommended forecaster and has not been since v11 of the ML model. Read it as a
record of an approach that was tried, measured and superseded, and read
`docs/ML_FORECAST_DESIGN.md` for what the measurements said.

WHY IT IS A PACKAGE AND NOT A DELETED FILE
------------------------------------------
Because the alternative loses the thing worth keeping. The statistical track is
the accuracy bar the whole ML project is judged against: every table in the
design document has a "prototype" column, and that column is this code. A reader
who wants to check a claim needs to be able to see what produced it.

MOUNTING IT
-----------
`router` is exported here so `api/main.py` can include it with one line if it is
ever wanted again. Whether it is currently mounted is decided in `api/main.py`,
not here. If it is not mounted, importing this package still costs nothing at
runtime beyond the import itself.
"""

from api.legacy.routes import router

__all__ = ["router"]
