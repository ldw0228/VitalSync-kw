# -*- coding: utf-8 -*-
"""창 이동을 5초 -> 1초로 줄여 데이터 확대 + 파형 정답까지 함께 저장
   저장: X(입력) · y_rate(호흡수) · y_wave(BIOPAC 파형) · base(고전 추정치)"""
import numpy as np, json, os
from scipy.signal import butter, filtfilt, welch
W='_work'; V=''
FSR=10.0; WIN=30.0; HOP=1.0; TT=int(WIN*FSR); TS=100; NB=8; BETA=0.98
BLK=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('회복 호흡',16,78),('운동 후 호흡',172,234)]
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def loopback(x,beta=BETA):
    c=np.empty_like(x); acc=x[0].copy()
    for n in range(len(x)): acc=beta*acc+(1-beta)*x[n]; c[n]=acc
    return x-c
def rate(x,fs):
    f,P=welch(x,fs=fs,nperseg=min(len(x),int(WIN*fs)),nfft=1<<14)
    m=(f>0.06)&(f<0.80); f,P=f[m],P[m]; i=int(np.argmax(P))
    if 0<i<len(P)-1:
        a,b,c=np.log(P[i-1]+1e-30),np.log(P[i]+1e-30),np.log(P[i+1]+1e-30)
        d=np.clip(0.5*(a-c)/(a-2*b+c+1e-30),-1,1); return float((f[i]+d*(f[1]-f[0]))*60)
    return float(f[i]*60)
def dsamp(a,n):
    idx=np.linspace(0,len(a)-1,n)
    if a.ndim==1: return np.interp(idx,np.arange(len(a)),a)
    return np.stack([np.interp(idx,np.arange(len(a)),a[:,k]) for k in range(a.shape[1])],1)
SY=json.load(open(V+'sync12_final.json'))
X=[];YR=[];YW=[];BS=[];SU=[];BL=[];TS_=[]
for o in SY:
    if o['grade']=='실패': continue
    s=o['s']; apr=o['ap']; apb=apr+o['off']
    pm=os.path.join(W,s+'_mot.npz'); pi=os.path.join(W,s+'_iq.npz')
    if not(os.path.exists(pm) and os.path.exists(pi)): continue
    z=np.load(pm); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    xb=bp(rsp,0.06,0.80,fsb)
    zi=np.load(pi); A=zi['re'].astype(np.float32)+1j*zi['im'].astype(np.float32)
    for nm,a0,a1 in BLK:
        j0,j1=int((apr+a0)*FSR),int((apr+a1)*FSR)
        if j0<0 or j1>A.shape[1]: continue
        chans=[]; proj=[]
        for r in range(3):
            cl=loopback(A[r,j0:j1])
            e=bp(np.abs(cl),0.1,0.6,FSR).std(0); k=int(np.argmax(e))
            b0=max(k-NB,0); b1=min(k+NB+1,A.shape[2])
            if b1-b0<2*NB+1: b0=max(min(b0,A.shape[2]-(2*NB+1)),0); b1=b0+2*NB+1
            sub=cl[:,b0:b1]; sc=np.abs(sub).std()+1e-9
            chans.append(np.concatenate([sub.real,sub.imag],1)/sc)
            c=sub[:,max(k-b0-1,0):k-b0+2].mean(1)
            xy=np.stack([c.real,c.imag],1); xy-=xy.mean(0)
            _,_,vt=np.linalg.svd(xy,full_matrices=False)
            proj.append(bp(xy@vt[0],0.06,0.80,FSR))
        seg=np.concatenate(chans,1)
        t=a0
        while t+WIN<=a1:
            j=int((t-a0)*FSR); w=seg[j:j+TT]
            ib0,ib1=int((apb+t)*fsb),int((apb+t+WIN)*fsb)
            if len(w)==TT and ib0>=0 and ib1<=len(xb):
                bw=xb[ib0:ib1]
                X.append(dsamp(w,TS).astype(np.float16))
                YR.append(rate(bw,fsb))
                yw=dsamp(bw,TS); YW.append((yw/(np.std(yw)+1e-9)).astype(np.float32))
                BS.append(float(np.median([rate(p[j:j+TT],FSR) for p in proj])))
                SU.append(s); BL.append(nm); TS_.append(t)
            t+=HOP
X=np.stack(X); YR=np.array(YR,np.float32); YW=np.stack(YW); BS=np.array(BS,np.float32)
np.savez_compressed(V+'ds2.npz',X=X,y_rate=YR,y_wave=YW,base=BS,
                    subj=np.array(SU),blk=np.array(BL),t=np.array(TS_,np.float32))
e=np.abs(BS-YR)
print('창 %d개 · 피험자 %d명 · 입력'%(len(YR),len(set(SU))),X.shape)
print('고전 기준선 (이 창들에서)  MAE %.2f  1BPM %.0f%%'%(e.mean(),100*np.mean(e<=1)))
