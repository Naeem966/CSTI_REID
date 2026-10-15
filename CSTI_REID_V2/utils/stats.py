"""Run-wise statistical reporting; no p-values from aggregates."""
import numpy as np
from scipy.stats import t

def summary(values):
    a=np.asarray(values,dtype=float)
    if len(a)<2:raise ValueError('Need >=2 runs for standard deviation')
    return {'n':len(a),'mean':float(a.mean()),'std':float(a.std(ddof=1))}

def paired(a,b):
    if set(a)!=set(b) or len(a)<2:raise ValueError('Require >=2 matched seed IDs')
    d=np.asarray([a[s]-b[s] for s in sorted(a)])
    mean=float(d.mean());sem=d.std(ddof=1)/np.sqrt(len(d))
    half=t.ppf(.975,len(d)-1)*sem
    return {'n':len(d),'delta':mean,'delta_sd':float(d.std(ddof=1)),
            'ci95_t':[float(mean-half),float(mean+half)]}
