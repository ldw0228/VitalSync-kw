# -*- coding: utf-8 -*-
"""레이더 호흡파형 ↔ BIOPAC RSP 상호상관으로 offset 추정"""
import numpy as np, os
from scipy.signal import butter, filtfilt, correlate, hilbert
from scipy.ndimage import uniform_filter1d
W='_work'
FSR=10.0

def bp(x, lo, hi, fs, order=3):
    b,a=butter(order,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)

def radar_resp(A, i0, i1):
    """A:(L,nB) complex.  가슴 bin 찾아 주성분 투영으로 1D 호흡 파형"""
    seg=A[i0:i1]
    e=np.abs(bp(np.abs(seg),0.12,0.6,FSR)).std(axis=0)
    kb=int(np.argmax(e))
    c=seg[:,max(kb-1,0):kb+2].mean(axis=1)
    xy=np.stack([c.real,c.imag],1); xy=xy-xy.mean(0)
    u,s,vt=np.linalg.svd(xy,full_matrices=False)
    w=(xy@vt[0])                                     # 주성분 투영
    return bp(w,0.12,0.6,FSR), kb, float(s[0]/(s.sum()+1e-9))

def env(x, fs=FSR, sm=8.0):
    return uniform_filter1d(np.abs(hilbert(x)), int(sm*fs))

def estimate(rad, bio, max_lag=25.0, fs=FSR):
    """2단계: 포락선으로 조대 정렬 → 파형으로 미세 정렬"""
    n=min(len(rad),len(bio)); rad=rad[:n]; bio=bio[:n]
    ml=int(max_lag*fs)
    def xc(a,b):
        a=(a-a.mean())/(a.std()+1e-9); b=(b-b.mean())/(b.std()+1e-9)
        c=correlate(a,b,mode='full')/len(a)
        lags=np.arange(-len(b)+1,len(a))
        m=np.abs(lags)<=ml
        return lags[m]/fs, c[m]
    lg,c=xc(env(rad),env(bio)); coarse=float(lg[int(np.argmax(c))])
    lg2,c2=xc(rad,bio)
    near=np.abs(lg2-coarse)<=2.5
    if not near.any(): return coarse, coarse, 0.0, 0.0
    idx=np.where(near)[0]; j=idx[int(np.argmax(np.abs(c2[idx])))]
    fine=float(lg2[j]); peak=float(np.abs(c2[j]))
    off=np.abs(lg2-fine)>1.0
    conf=float((np.abs(c2[j])-np.abs(c2[off]).max())/(np.abs(c2).std()+1e-9)) if off.any() else 0.0
    return coarse, fine, peak, conf
