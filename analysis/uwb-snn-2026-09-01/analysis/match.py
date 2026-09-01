# -*- coding: utf-8 -*-
"""서브젝트별 호흡 복원 일치도 — 레이더 BPM vs BIOPAC BPM"""
import numpy as np, json, os, glob
from scipy.signal import butter, filtfilt, welch
from scipy.ndimage import uniform_filter1d
W='_work'
AN=json.load(open('anchors.json'))
FSR=10.0
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def bpm(x,fs):
    n=min(len(x),int(51*fs))
    f,P=welch(x,fs=fs,nperseg=n)
    m=(f>0.08)&(f<0.7)
    return float(f[m][np.argmax(P[m])]*60)
def apnea_c(env,fsb,lo,hi):
    k=int(22*fsb); ma=uniform_filter1d(env,k)
    i0=max(int(lo*fsb),k//2); i1=min(int(hi*fsb),len(ma)-k//2)
    if i1-i0<k: return None
    i=int(np.argmin(ma[i0:i1]))+i0
    return i/fsb, float(ma[i]/(np.median(ma)+1e-12))
def radar_wave(A,r,j0,j1,ref=None):
    s=A[r,j0:j1].copy(); s=s-s.mean(0,keepdims=True)
    e=bp(np.abs(s),0.1,0.6,FSR).std(0); k=int(np.argmax(e)) if ref is None else ref
    c=s[:,max(k-1,0):k+2].mean(1)
    xy=np.stack([c.real,c.imag],1); xy=xy-xy.mean(0)
    _,_,vt=np.linalg.svd(xy,full_matrices=False)
    return bp(xy@vt[0],0.08,0.7,FSR), k
BLK=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('운동 후 호흡',172,234)]
res={}
for s,v in sorted(AN.items()):
    pm=os.path.join(W,s+'_mot.npz'); pi=os.path.join(W,s+'_iq.npz')
    if not(os.path.exists(pm) and os.path.exists(pi)): continue
    z=np.load(pm); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    xb=bp(rsp,0.08,0.8,fsb); env=uniform_filter1d(np.abs(xb),int(4*fsb))
    r0=apnea_c(env,fsb,max(v['ex_center']-260,20),v['ex_center']-40)
    if r0 is None or r0[1]>0.5: continue
    apb=r0[0]                       # BIOPAC 시간축 무호흡 중심
    zi=np.load(pi); A=zi['re'].astype(np.float32)+1j*zi['im'].astype(np.float32)
    apr=v['ap_center']              # 레이더 시간축 무호흡 중심
    rows=[]
    for nm,a0,a1 in BLK:
        ib0,ib1=int((apb+a0)*fsb),int((apb+a1)*fsb)
        jr0,jr1=int((apr+a0)*FSR),int((apr+a1)*FSR)
        if ib0<0 or ib1>len(xb) or jr0<0 or jr1>A.shape[1]: continue
        bb=bpm(xb[ib0:ib1],fsb)
        rb=[]
        for r in range(A.shape[0]):
            w,_=radar_wave(A,r,jr0,jr1); rb.append(bpm(w,FSR))
        rows.append((nm,bb,rb))
    if len(rows)<2: continue
    err=[]
    for nm,bb,rb in rows: err += [abs(x-bb) for x in rb]
    res[s]=dict(rows=[(nm,bb,rb) for nm,bb,rb in rows],
                mae=float(np.median(err)), best=float(np.min(err)), apb=apb, apr=apr)
order=sorted(res,key=lambda k:res[k]['mae'])
print('%-10s %8s   %s'%('피험자','BPM 오차(중앙값)','블록별 BIOPAC / 레이더1,2,3'))
for s in order:
    d=res[s]
    print('%-10s %8.2f   '%(s,d['mae']),end='')
    print(' | '.join('%s %.1f / %s'%(nm,bb,','.join('%.1f'%x for x in rb)) for nm,bb,rb in d['rows']))
m=np.array([res[s]['mae'] for s in order])
print('\nn=%d  중앙값 %.2f BPM  |  1.0 이하 %d명  |  2.5 초과 %d명'%(
    len(m),np.median(m),(m<=1.0).sum(),(m>2.5).sum()))
json.dump({k:{kk:vv for kk,vv in v.items()} for k,v in res.items()},
          open('match.json','w'),ensure_ascii=False)
