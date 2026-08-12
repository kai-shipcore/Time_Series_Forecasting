import sys; from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import pandas as pd
from src.ml.dataset import dev_splits, load_weekly, stratified_val_skus
from src.ml.evaluate import bootstrap_delta, is_significant, score, score_table
from src.ml.model import FEATURES_V1, RatioLGBM, structural_baseline

d=pd.read_csv("outputs/reports/tune_wide.csv").sort_values("val_l1")
win=d.iloc[0]
WIN=dict(learning_rate=float(win.learning_rate),num_leaves=int(win.num_leaves),
         min_child_samples=int(win.min_child_samples),
         colsample_bytree=float(win.colsample_bytree),subsample=float(win.subsample),
         reg_alpha=float(win.reg_alpha),reg_lambda=float(win.reg_lambda))
PAT=int(win.patience)
print(f"winner {win.tag}: {WIN}, patience={PAT}\n")

w,p=load_weekly(); sm=p.loc[p.bucket=="smooth","unique_id"]; w=w[w.unique_id.isin(set(sm))]

# Reference figures come from src/ml/reference.py, which carries the snapshot
# they were measured on and warns when it is not the active one. This file used
# to hold its own copy, the last of seven such copies, and it was still on the
# 2026-07-20 values.
from src.ml.reference import PROTOTYPE as PROTO, warn_if_stale  # noqa: E402
warn_if_stale()

def fit(s,val,params,pat):
    # params MUST go through the constructor argument. This function previously
    # assigned `m.PARAMS` after construction, which today does nothing: __init__
    # has already read self.PARAMS into self.params, and fit() builds the
    # estimator from self.params. Left alone, every arm would silently fit the
    # DEFAULT configuration while reporting the tuned one.
    #
    # A LATENT REGRESSION, not a historical error, and the distinction was
    # checked rather than assumed. The `params=` argument landed 2026-07-29;
    # these scripts last ran 2026-07-21. When they ran, the attribute was the
    # supported route and it worked: outputs/reports/tune_wide.csv holds 81
    # configurations with 81 distinct val_l1 scores, which is impossible if the
    # hyperparameters had not reached the estimator, since only `patience` is
    # passed through the constructor. So v10's rejection stands.
    #
    # What the refactor broke is any FUTURE run. Its comment reads "Default None
    # reproduces PARAMS exactly, so existing versions are unaffected", which is
    # true of everything using the argument and false of the three tuning
    # scripts, which used the attribute. ml_17 and ml_18 carry the same fix.
    # See docs/ML_FORECAST_DESIGN.md Section 4.33.
    pr=dict(params) if params else None
    if pr and pr.get("subsample",1.0)<1.0: pr["subsample_freq"]=1
    m=RatioLGBM(s.horizon,FEATURES_V1,deseas_features=True,deseas_all=True,
                patience=pat,params=pr)
    assert not params or m.params["num_leaves"]==params.get("num_leaves",m.params["num_leaves"]), \
        "tuned params did not reach the estimator"
    return m.fit(s.train,p,s.cutoff,val)
for s,n in zip(dev_splits(w,n=3),PROTO):
    val=stratified_val_skus(s.train,p)
    p9=fit(s,val,None,100).predict(s.train,p,s.cutoff)
    p10=fit(s,val,WIN,PAT).predict(s.train,p,s.cutoff)
    res={"baseline":score(structural_baseline(s.train,s.test,p,s.cutoff),s,p),
         "v9":score(p9,s,p),"v10":score(p10,s,p)}
    print(f"{'='*56}\n{n}   prototype short={PROTO[n][0]}, long={PROTO[n][1]}")
    print(score_table(res).to_string())
    for seg in ("short","long"):
        bd=bootstrap_delta(p10,p9,s,p,segment=seg)
        print(f"  v10-vs-v9 [{seg}]: delta={bd['delta']:+.4f} se={bd['se']:.4f} sig={is_significant(bd)}")
