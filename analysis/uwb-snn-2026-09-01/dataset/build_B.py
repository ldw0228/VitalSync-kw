# -*- coding: utf-8 -*-
"""과제 B(호흡 상태) 데이터셋 — 레이더 무호흡을 기준점으로 정박"""
import numpy as np, os, glob, json
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
W='_work'
LM=json.load(open('landmarks.json')); AN=json.load(open('anchors.json'))
FSR=10.0; DS=2; FS=FSR/DS; WIN_S=20.0; T=int(WIN_S*FS); NBIN=8
# 무호흡 중심 기준 상대 구간 (초). 대본 375초 환산, 15초 버퍼 2개는 제외
BLOCKS=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('무호흡',-16,16),
        ('회복 호흡',16,78),('운동',94,156),('운동 후 호흡',172,234)]
NAMES=[b[0] for b in BLOCKS]
TARGET=[10,10,14,10,10,10]
def bp(x,lo,hi,fs=FSR,order=3):
    b,a=butter(order,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def radar_apnea(A, ex_center):
    lo=int((ex_center-260)*FSR); hi=min(int((ex_center-40)*FSR),A.shape[1])
    if lo<0 or hi-lo<int(120*FSR): return None
    envs=[]
    for r in range(A.shape[0]):
        seg=A[r,lo:hi]-A[r,lo:hi].mean(0,keepdims=True)
        e=bp(np.abs(seg),0.1,0.6).std(axis=0); kb=int(np.argmax(e))
        c=seg[:,max(kb-1,0):kb+2].mean(1)
        xy=np.stack([c.real,c.imag],1); xy-=xy.mean(0)
        _,_,vt=np.linalg.svd(xy,full_matrices=False)
        envs.append(uniform_filter1d(np.abs(bp(xy@vt[0],0.1,0.6)),int(4*FSR)))
    E=np.mean(envs,0); t=np.arange(len(E))/FSR+(ex_center-260)
    k=int(20*FSR); ma=uniform_filter1d(E,k)
    i=int(np.argmin(ma[k//2:len(ma)-k//2]))+k//2
    return float(t[i]), float(ma[i]/(np.median(ma)+1e-12))
X=[];y=[];g=[];rad=[]
for p in sorted(glob.glob(os.path.join(W,'*_iq.npz'))):
    subj=os.path.basename(p)[:-7]
    if subj not in AN or subj not in LM: continue
    z=np.load(p); A=z['re'].astype(np.float32)+1j*z['im'].astype(np.float32)
    ra=radar_apnea(A, AN[subj]['ex_center'])
    if ra is None: continue
    ap,dep=ra
    if dep>0.5: continue
    # 1번 구간에서 bin/방향/스케일 고정 (세션 보정)
    bl=LM[subj]['blocks']; lo1=max(int(bl[0][0]*FSR),0); hi1=min(int(bl[2][1]*FSR),A.shape[1])
    for r in range(A.shape[0]):
        base=A[r,lo1:hi1]-A[r,lo1:hi1].mean(0,keepdims=True)
        e=bp(np.abs(base),0.1,0.6).std(axis=0); kb=int(np.argmax(e))
        b0=max(kb-NBIN,0); b1=min(kb+NBIN+1,A.shape[2])
        if b1-b0<2*NBIN+1: b0=max(min(b0,A.shape[2]-(2*NBIN+1)),0); b1=b0+2*NBIN+1
        sub=base[:,b0:b1]
        dirs=[]
        for k in range(sub.shape[1]):
            xy=np.stack([sub[:,k].real,sub[:,k].imag],1); xy-=xy.mean(0)
            _,_,vt=np.linalg.svd(xy,full_matrices=False); dirs.append(vt[0])
        dirs=np.array(dirs)
        pa=np.stack([(np.stack([sub[:,k].real,sub[:,k].imag],1)@dirs[k]) for k in range(sub.shape[1])],1)
        scale=float(bp(pa,0.1,0.6).std())+1e-9
        for ci,(nm,a0,a1) in enumerate(BLOCKS):
            j0=int((ap+a0)*FSR); j1=int((ap+a1)*FSR)
            if j0<0 or j1>A.shape[1] or j1-j0<int(WIN_S*FSR)+10: continue
            s=A[r,j0:j1].copy(); s=s-s.mean(0,keepdims=True); s=s[:,b0:b1]
            proj=np.stack([(np.stack([s[:,k].real,s[:,k].imag],1)@dirs[k]) for k in range(s.shape[1])],1)
            proj=bp(proj,0.08,0.8)/scale
            mag=bp(np.abs(s),0.08,0.8)/scale
            F=np.concatenate([proj,mag],1)
            F=F[:(len(F)//DS)*DS].reshape(-1,DS,F.shape[1]).mean(1)
            span=len(F)-T
            if span<=0: continue
            n=TARGET[ci]; stride=max(1,span//max(n-1,1))
            for st in range(0,span+1,stride):
                X.append(F[st:st+T].astype(np.float16)); y.append(ci); g.append(subj); rad.append(r+1)
X=np.stack(X);y=np.array(y);g=np.array(g);rad=np.array(rad)
print('X',X.shape,'| 피험자',len(set(g.tolist())))
import collections
print('클래스:',{NAMES[k]:v for k,v in sorted(collections.Counter(y).items())})
np.savez_compressed('dataset_B.npz',X=X,y=y,g=g,rad=rad,names=np.array(NAMES))
