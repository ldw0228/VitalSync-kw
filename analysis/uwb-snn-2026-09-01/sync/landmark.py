# -*- coding: utf-8 -*-
"""1번 구간의 회전 2회를 레이더 움직임에서 찾는다.
   시작마커·회전1·회전2·종료마커가 모두 ~62초 간격이므로,
   '회전1은 녹화 시작 후 50~120초'라는 제약으로 역할을 확정한다."""
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
FPS=40.0
P1_LO, P1_HI = 50.0, 120.0
def rotations(mot, tmax=400.0, gap_lo=57.0, gap_hi=68.0):
    s=uniform_filter1d(mot.mean(0).astype(float), int(1.5*FPS))
    n=min(int(tmax*FPS), len(s)); seg=s[:n]; t=np.arange(n)/FPS
    base=np.median(seg)+1e-12
    pk,_=find_peaks(seg, height=base*1.4, distance=int(12*FPS), prominence=base*0.5)
    if len(pk)<2: return None
    cands=[]
    for i in range(len(pk)):
        if not (P1_LO<=t[pk[i]]<=P1_HI): continue
        for j in range(i+1,len(pk)):
            g=t[pk[j]]-t[pk[i]]
            if not (gap_lo<=g<=gap_hi): continue
            a,b=pk[i],pk[j]
            quiet=np.median(seg[a:b])+1e-12
            score=(seg[a]+seg[b])/quiet
            pre=max(a-int(55*FPS),0)
            if pre<a: score*=(seg[a]/(np.median(seg[pre:a])+1e-12))**0.5
            cands.append((score,t[pk[i]],t[pk[j]],g))
    if not cands: return None
    cands.sort(key=lambda x:-x[0]); sc,p1,p2,g=cands[0]
    return dict(p1=float(p1),p2=float(p2),gap=float(g),score=float(sc))

def blocks(p1, p2, guard=3.0, span=58.0):
    """각도 블록 (레이더 시간).  0deg / 45deg / 90deg"""
    return [(p1-guard-span, p1-guard), (p1+guard, p2-guard), (p2+guard, p2+guard+span)]
