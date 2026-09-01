# -*- coding: utf-8 -*-
import numpy as np, json, os
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
W='_work'
LM=json.load(open('landmarks.json')); AN=json.load(open('anchors.json')); FSR=10.0
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def env_radar(A,r):
    seg=A[r]-A[r].mean(0,keepdims=True)
    e=bp(np.abs(seg),0.1,0.6,FSR).std(0); kb=int(np.argmax(e))
    c=seg[:,max(kb-1,0):kb+2].mean(1)
    xy=np.stack([c.real,c.imag],1); xy-=xy.mean(0)
    _,_,vt=np.linalg.svd(xy,full_matrices=False)
    return uniform_filter1d(np.abs(bp(xy@vt[0],0.1,0.6,FSR)),int(4*FSR)), kb+8
def dip(e,lo,hi,win=20.0):
    k=int(win*FSR); ma=uniform_filter1d(e,k)
    i0=max(int(lo*FSR),k//2); i1=min(int(hi*FSR),len(ma)-k//2)
    if i1-i0<k: return None
    i=int(np.argmin(ma[i0:i1]))+i0
    return i/FSR, float(ma[i]/(np.median(ma)+1e-12))
for s in ['S15_JKH','S23_KDM','S10_JKH','S18_LJH']:
    end1=LM[s]['blocks'][2][1]
    z=np.load(os.path.join(W,s+'_iq.npz'))
    A=z['re'].astype(np.float32)+1j*z['im'].astype(np.float32)
    old=AN.get(s,{}).get('ap_center')
    print('%s  end1 %.1f  기존 anchors ap=%s  탐색창 %.0f~%.0f'%(
        s,end1,('%.1f'%old) if old else '없음',end1+100,end1+260))
    for r in range(3):
        e,kb=env_radar(A,r)
        d=dip(e,end1+100,end1+260)
        d2=dip(e,end1+60,end1+320)          # 더 넓게
        print('   r%d bin %2d | 창내 %.1f초 함몰 %.2f | 넓게 %.1f초 함몰 %.2f'%(
            r+1,kb,d[0],d[1],d2[0],d2[1]))
