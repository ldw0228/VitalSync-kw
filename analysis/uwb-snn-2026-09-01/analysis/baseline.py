# -*- coding: utf-8 -*-
"""기준선 확정 — 실험 1·2 정박(28명) · 연속 호흡수 추정 · 레이더 3대 융합"""
import numpy as np, json, os
from scipy.signal import butter, filtfilt, welch
W='_work'
SY=json.load(open('sync12_final.json')); FSR=10.0
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def bpm(x,fs):
    """포물선 보간으로 FFT 격자보다 미세한 주파수 추정"""
    n=len(x); f,P=welch(x,fs=fs,nperseg=min(n,int(51*fs)),nfft=1<<14)
    m=(f>0.08)&(f<0.75); f=f[m]; P=P[m]
    i=int(np.argmax(P))
    if 0<i<len(P)-1:                      # 로그 스펙트럼 3점 포물선 정점
        a,b,c=np.log(P[i-1]+1e-30),np.log(P[i]+1e-30),np.log(P[i+1]+1e-30)
        d=0.5*(a-c)/(a-2*b+c+1e-30); d=np.clip(d,-1,1)
        return float((f[i]+d*(f[1]-f[0]))*60)
    return float(f[i]*60)
def radar_bpm(A,r,j0,j1):
    s=A[r,j0:j1].copy(); s=s-s.mean(0,keepdims=True)
    e=bp(np.abs(s),0.1,0.6,FSR).std(0); k=int(np.argmax(e))
    c=s[:,max(k-1,0):k+2].mean(1)
    xy=np.stack([c.real,c.imag],1); xy=xy-xy.mean(0)
    _,_,vt=np.linalg.svd(xy,full_matrices=False)
    return bpm(bp(xy@vt[0],0.08,0.75,FSR),FSR)
BLK=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('회복 호흡',18,46),('운동 후 호흡',172,234)]
rec=[]
for o in SY:
    if o['grade']=='실패': continue
    s=o['s']; apr=o['ap']; apb=apr+o['off']
    pm=os.path.join(W,s+'_mot.npz'); pi=os.path.join(W,s+'_iq.npz')
    if not(os.path.exists(pm) and os.path.exists(pi)): continue
    z=np.load(pm); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    xb=bp(rsp,0.08,0.75,fsb)
    zi=np.load(pi); A=zi['re'].astype(np.float32)+1j*zi['im'].astype(np.float32)
    for nm,a0,a1 in BLK:
        ib0,ib1=int((apb+a0)*fsb),int((apb+a1)*fsb)
        jr0,jr1=int((apr+a0)*FSR),int((apr+a1)*FSR)
        if ib0<0 or ib1>len(xb) or jr0<0 or jr1>A.shape[1]: continue
        bb=bpm(xb[ib0:ib1],fsb)
        rb=[radar_bpm(A,r,jr0,jr1) for r in range(A.shape[0])]
        rec.append(dict(s=s,blk=nm,bio=bb,rad=rb))
json.dump(rec,open('baseline.json','w'),ensure_ascii=False)
def stat(err):
    a=np.array(err); return a.mean(), np.median(a), 100*np.mean(a<=1.0), 100*np.mean(a<=2.5)
print('피험자 %d명 · 블록 %d개 · 총 %d건\n'%(len(set(r['s'] for r in rec)),len(BLK),len(rec)))
ways={'레이더1':lambda r:r['rad'][0],'레이더2':lambda r:r['rad'][1],'레이더3':lambda r:r['rad'][2],
      '3대 평균':lambda r:float(np.mean(r['rad'])),'3대 중앙값':lambda r:float(np.median(r['rad'])),
      '3대 최선(상한)':None}
print('%-16s %7s %7s %8s %8s'%('융합 방식','MAE','중앙값','≤1 BPM','≤2.5 BPM'))
for k,f in ways.items():
    if f is None: e=[min(abs(x-r['bio']) for x in r['rad']) for r in rec]
    else: e=[abs(f(r)-r['bio']) for r in rec]
    m,md,p1,p25=stat(e)
    print('%-16s %7.2f %7.2f %7.0f%% %7.0f%%'%(k,m,md,p1,p25))
print('\n블록별 (3대 중앙값 / 최선 상한)')
for nm,_,_ in BLK:
    sub=[r for r in rec if r['blk']==nm]
    em=[abs(float(np.median(r['rad']))-r['bio']) for r in sub]
    eb=[min(abs(x-r['bio']) for x in r['rad']) for r in sub]
    print('  %-12s n=%2d  MAE %5.2f → %5.2f   ≤1BPM %3.0f%% → %3.0f%%   BIOPAC 평균 %.1f BPM'%(
        nm,len(sub),np.mean(em),np.mean(eb),100*np.mean(np.array(em)<=1),100*np.mean(np.array(eb)<=1),
        np.mean([r['bio'] for r in sub])))
