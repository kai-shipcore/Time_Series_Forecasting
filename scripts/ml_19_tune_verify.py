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
PROTO={"Mar-May":(0.2014,0.1411),"Dec-Feb":(0.2863,0.2737),"Oct-Dec":(0.4251,0.0911)}
def fit(s,val,params,pat):
    m=RatioLGBM(s.horizon,FEATURES_V1,deseas_features=True,deseas_all=True,patience=pat)
    if params: m.PARAMS={**RatioLGBM.PARAMS,**params}
    if params and params.get("subsample",1.0)<1.0: m.PARAMS["subsample_freq"]=1
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
