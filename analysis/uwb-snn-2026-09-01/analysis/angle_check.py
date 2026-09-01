# -*- coding: utf-8 -*-
"""각도 물리 검증: 상대각에 따라 레이더별 호흡 변조가 달라지는가"""
import numpy as np, os, glob, json
from scipy.signal import butter, filtfilt, welch
W='_work'
FSR=10.0
LM=json.load(open('landmarks.json'))
# |상대각| 표 : ABS[radar_index][block_index]
ABS=np.array([[ 0,45,90],
              [45, 0,45],
              [90,45, 0]])

def bp(x,lo,hi,fs=FSR,order=3):
    b,a=butter(order,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)

def chest_bin(A, i0, i1):
    seg=A[i0:i1]-A[i0:i1].mean(0,keepdims=True)
    e=bp(np.abs(seg),0.1,0.6).std(axis=0)
    return int(np.argmax(e))

def block_feat(A, kb, i0, i1):
    seg=A[i0:i1, max(kb-1,0):kb+2]
    seg=seg-seg.mean(0,keepdims=True)
    c=seg.mean(axis=1)
    xy=np.stack([c.real,c.imag],1); xy-=xy.mean(0)
    _,s,vt=np.linalg.svd(xy,full_matrices=False)
    w=bp(xy@vt[0],0.1,0.6)
    amp=float(np.std(w))
    # 호흡 대역 SNR
    f,P=welch(w,FSR,nperseg=min(512,len(w)))
    band=(f>0.1)&(f<0.6); noise=(f>0.8)&(f<2.0)
    snr=float(P[band].sum()/(P[noise].sum()+1e-12))
    mag=bp(np.abs(seg).mean(axis=1),0.1,0.6)
    return dict(amp=amp, snr=snr, mag_amp=float(np.std(mag)), pc_ratio=float(s[0]/s.sum()))
