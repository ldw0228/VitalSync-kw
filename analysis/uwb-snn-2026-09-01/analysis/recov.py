# -*- coding: utf-8 -*-
"""무호흡 종료 이후 호흡 진폭/호흡률 회복 곡선 (BIOPAC 기준)"""
import numpy as np, json, os
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d
W='_work'
AN=json.load(open('anchors.json'))
def bp(x,lo,hi,fs,order=3):
    b,a=butter(order,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x)
def bio_apnea(env,fsb,lo,hi):
    """[lo,hi]초 안에서 22초 이동평균 포락선이 가장 낮은 지점 = 무호흡 중심"""
    k=int(22*fsb); ma=uniform_filter1d(env,k)
    i0=max(int(lo*fsb),k//2); i1=min(int(hi*fsb),len(ma)-k//2)
    if i1-i0<k: return None
    i=int(np.argmin(ma[i0:i1]))+i0
    return i/fsb, float(ma[i]/(np.median(ma)+1e-12))
BINS=[(-140,-80),(0,20),(20,40),(40,60),(60,80)]
LBL=['평소호흡(기준)','무호흡후 0~20초','20~40초','40~60초','60~80초']
amp={i:[] for i in range(len(BINS))}; rate={i:[] for i in range(len(BINS))}; used=[]
for s,v in sorted(AN.items()):
    p=os.path.join(W,s+'_mot.npz')
    if not os.path.exists(p): continue
    z=np.load(p); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    x=bp(rsp,0.08,0.8,fsb); env=uniform_filter1d(np.abs(x),int(4*fsb))
    exc=v['ex_center']
    r=bio_apnea(env,fsb,max(exc-260,20),exc-40)
    if r is None: continue
    apc,dep=r
    if dep>0.5: continue
    ape=apc+15.0
    A=[];R=[];ok=True
    for (a0,a1) in BINS:
        i0=int((ape+a0)*fsb); i1=int((ape+a1)*fsb)
        if i0<0 or i1>len(x): ok=False; break
        A.append(float(np.median(env[i0:i1])))
        seg=x[i0:i1]
        pk,_=find_peaks(seg,distance=int(1.2*fsb),prominence=np.std(seg)*0.5)
        R.append(len(pk)/((a1-a0)/60.0))
    if not ok or A[0]<=0: continue
    used.append(s)
    for i in range(len(BINS)):
        amp[i].append(A[i]/A[0]); rate[i].append(R[i]-R[0])
print('피험자 %d명: %s\n'%(len(used),', '.join(used)))
out={}
for i,l in enumerate(LBL):
    a=np.array(amp[i]); r=np.array(rate[i])
    n_up=int((a>1.15).sum())
    out[l]=dict(amp_med=round(float(np.median(a)),3),amp_mean=round(float(np.mean(a)),3),
                amp_sd=round(float(np.std(a)),3),rate_med=round(float(np.median(r)),2),
                n_amp_up=n_up,n=len(a))
    print('%-14s 진폭비 %.2f (평균 %.2f ±%.2f) | 호흡률차 %+.1f/min | 진폭 15%%↑ %d/%d명'%(
        l,np.median(a),np.mean(a),np.std(a),np.median(r),n_up,len(a)))
json.dump(dict(subjects=used,stats=out),open('recovery.json','w'),ensure_ascii=False,indent=1)
