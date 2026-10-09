"""Known RRC pulse coefficients for optional pilot timing on pulse-shaped data."""
from __future__ import annotations
import math
import numpy as np

def rrc_taps(beta: float, span_symbols: int, sps: int) -> np.ndarray:
    n = int(span_symbols) * int(sps)
    if n % 2: n += 1
    t = np.arange(-n/2, n/2+1, dtype=float) / float(sps)
    h=np.zeros_like(t); beta=float(beta)
    for i,x in enumerate(t):
        if abs(x)<1e-12: h[i]=1+beta*(4/math.pi-1)
        elif beta>0 and abs(abs(x)-1/(4*beta))<1e-10:
            h[i]=(beta/math.sqrt(2))*((1+2/math.pi)*math.sin(math.pi/(4*beta))+(1-2/math.pi)*math.cos(math.pi/(4*beta)))
        else:
            den=math.pi*x*(1-(4*beta*x)**2)
            h[i]=(math.sin(math.pi*x*(1-beta))+4*beta*x*math.cos(math.pi*x*(1+beta)))/den if abs(den)>1e-14 else 0.0
    return h/np.sqrt(np.sum(h*h))


