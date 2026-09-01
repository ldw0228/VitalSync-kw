# -*- coding: utf-8 -*-
"""블록별 BIOPAC 호흡률 분포 — 호흡수 추정 과제의 목표 범위가 얼마나 넓은지 확인"""
import numpy as np, json, os
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d
W='_work'
AN=json.load(open('anchors.json'))
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x)
def apnea_c(env,fsb,lo,hi):
    k=int(22*fsb); ma=uniform_filter1d(env,k)
    i0=max(int(lo*fsb),k//2); i1=min(int(hi*fsb),len(ma)-k//2)
    if i1-i0<k: return None
    i=int(np.argmin(ma[i0:i1]))+i0
    return i/fsb, float(ma[i]/(np.median(ma)+1e-12))
BLOCKS=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('회복 호흡',16,78),
        ('운동 후 호흡',172,234)]
res={n:[] for n,_,_ in BLOCKS}
for s,v in sorted(AN.items()):
    p=os.path.join(W,s+'_mot.npz')
    if not os.path.exists(p): continue
    z=np.load(p); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    x=bp(rsp,0.08,0.8,fsb); env=uniform_filter1d(np.abs(x),int(4*fsb))
    r=apnea_c(env,fsb,max(v['ex_center']-260,20),v['ex_center']-40)
    if r is None or r[1]>0.5: continue
    ap=r[0]
    for nm,a0,a1 in BLOCKS:
        i0=int((ap+a0)*fsb); i1=int((ap+a1)*fsb)
        if i0<0 or i1>len(x): continue
        seg=x[i0:i1]
        pk,_=find_peaks(seg,distance=int(1.2*fsb),prominence=np.std(seg)*0.5)
        res[nm].append(len(pk)/((a1-a0)/60.0))
print('%-14s %6s %6s %6s %6s  %s'%('블록','최소','25%','중앙','75%','최대'))
allv=[]
for nm,_,_ in BLOCKS:
    a=np.array(res[nm]); allv+=list(a)
    q=np.percentile(a,[0,25,50,75,100])
    print('%-14s %6.1f %6.1f %6.1f %6.1f %6.1f   (n=%d)'%(nm,*q,len(a)))
a=np.array(allv)
print('\n전체 범위 %.1f ~ %.1f 회/분, 표준편차 %.1f'%(a.min(),a.max(),a.std()))
