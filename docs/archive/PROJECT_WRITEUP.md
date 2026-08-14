# Interview study notes

Personal preparation notes for explaining the project. Not a reference document;
the references are `OVERVIEW.md`, `MODEL.md`, `DATA_AND_PIPELINE.md` and `SCREENS.md`.

---

## The project in one paragraph

The company forecasts weekly demand for vehicle accessories so it can order the right amount
of stock. The old method is a spreadsheet formula that blends recent sales and multiplies by
hand-set seasonal factors; it systematically forecasts wrong because trailing averages cannot
anticipate change. We built a machine-learning model that forecasts a demand multiplier on
top of each product's recent level, learns patterns from the whole catalog at once, and
handles seasonality structurally rather than learning it from two years of data. It beats the
spreadsheet in four of six seasonal tests, often by a wide margin, and ties a plain moving
average in flat quarters while significantly beating it at seasonal turning points.

---

## What I would want to explain on the spot

1. **Why predict a multiplier, and what "predicting 1.0" reproduces.** A model that learns
   nothing predicts 1.0 and reproduces a twelve-week moving average exactly. That is the
   floor. Every point of accuracy above it is attributable to the model, which is what makes
   comparison honest.

2. **Pooled WAPE on ten-week totals, and why that matches how the forecast is used.** Stock
   is ordered for a horizon, not for each week separately, and a miss on a big seller costs
   more than on a small one.

3. **The quarantined-test discipline and why it matters.** Three development windows spanning
   three seasonal regimes so a change has to work across seasons. One window quarantined and
   never looked at during development, used once as the final go/no-go. The value is that the
   criteria were fixed before it ran.

4. **The v4-to-v11 story in three sentences.** The segments want opposite things from a
   shared model: changes that helped established products damaged newer ones through shared
   trees. The real cause was a mis-specified seasonal calendar plus a missing turning-point
   signal (elevation). Splitting the model and adding elevation fixed the hardest cell without
   touching the others.

5. **Why the model is near the ceiling on the sales record.** Trees interpolate well and
   extrapolate terribly. The remaining error lives in low and mid-volume SKUs where the model
   cannot be shown to beat a simple average. Further gains need better input data (stockout
   correction, preorder attribution), not more features.

6. **Where the model actually earns its keep.** Not everywhere uniformly. At the two
   seasonal turning points (Q4 ramp-up, post-holiday decline) it cuts error roughly in half
   versus a trailing average. In flat quarters they are the same forecaster. The value is
   concentrated, which is honest to say and more interesting than a single headline number.

7. **What was rejected and why it matters.** Six perturbations of the long model each cost it
   half a point to a point. That list is a better account of the work than the changes that
   landed, because it shows a model at a narrow optimum on a small sample where nearly any
   added degree of freedom costs more in variance than it returns in signal.

---

## The version story as a narrative

- **v0-v3** built the model up one idea at a time: growth-drift correction, trajectory
  features, full seasonal consistency. v3 was strong everywhere except one cell: established
  products after the holidays, which it over-forecast.
- **v4, v5, v7, v8** all tried to fix that cell by treating it as a segment-handling problem:
  a segment flag, per-segment seasonal factors, per-segment weights, fully separate models.
  Every one improved established products and damaged newer ones. The repeated failure was the
  clue: the segments genuinely want opposite things from a shared model.
- **v6/v9** found the real bug. The holiday uplift was applied to late December, when demand
  had already fallen, and the calendar window drifted year to year. Fixing it to match the
  real promotional period helped a lot.
- **v10** tried hyperparameter tuning and confirmed the remaining gap was not a tuning
  problem.
- **v11** solved it. The post-holiday over-forecast was the model extrapolating a ramp into a
  decline with no feature that could see the turn. Elevation is the missing signal. Putting it
  in a dedicated long-product model finally beat the moving-average floor in the cell that had
  blocked every version.
- **v12-v18** were all rejected, confirming the ceiling on sales data alone.

The one-sentence version: it was never a "which group" problem and never a tuning problem; it
was a mis-specified seasonal calendar plus a missing turning-point signal.
