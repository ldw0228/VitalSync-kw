# -*- coding: utf-8 -*-
"""2번 세부 라벨: 레이더에서 운동 블록 검출 + BIOPAC 호흡률로 교차 검증"""
import numpy as np, os, json
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks
FPS=40.0
def breath_rate(rsp, fsb, t0, t1, win=30.0, hop=5.0):
    """구간별 호흡률(회/분)과 진폭"""
    b,a=butter(3,[0.08/(fsb/2),0.7/(fsb/2)],btype='band'); x=filtfilt(b,a,rsp)
    ts=[];rates=[];amps=[]
    t=t0
    while t+win<=t1:
        i0,i1=int(t*fsb),int((t+win)*fsb)
        if i1>len(x): break
        seg=x[i0:i1]
        pk,_=find_peaks(seg, distance=int(1.5*fsb), prominence=np.std(seg)*0.5)
        rates.append(len(pk)/win*60.0); amps.append(float(np.std(seg))); ts.append(t+win/2)
        t+=hop
    return np.array(ts),np.array(rates),np.array(amps)

def find_exercise(mot, end1, lo_off=245.0, hi_off=395.0, dur=60.0):
    s=uniform_filter1d(mot.mean(0).astype(float), int(3*FPS))
    lo=int((end1+lo_off)*FPS); hi=min(int((end1+hi_off)*FPS),len(s))
    if hi-lo<int(dur*FPS): return None
    k=int(dur*FPS); ma=uniform_filter1d(s,k)
    win=ma[lo:hi]; t=np.arange(lo,hi)/FPS
    c=int(np.argmax(win)); ctr=float(t[c])
    # 직후 60초가 조용해야 함 (운동 후 호흡)
    p0,p1=int((ctr+35)*FPS),int((ctr+95)*FPS)
    post=float(np.median(s[p0:p1])) if p1<=len(s) else np.nan
    return dict(center=ctr, start=ctr-dur/2, end=ctr+dur/2,
                contrast=float(win[c]/(np.median(s[lo:hi])+1e-12)),
                post_ratio=float(win[c]/(post+1e-12)) if post==post else None)
