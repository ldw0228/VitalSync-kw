# -*- coding: utf-8 -*-
"""올바른 파싱 후에도 남는 2차 봉우리가 진짜 호흡을 담고 있는가"""
import numpy as np, glob, os, json
from scipy.signal import butter, filtfilt, welch
W='_work'
def bp(x,lo,hi,fs=10.0,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def f0(x,fs=10.0):
    f,P=welch(x,fs=fs,nperseg=min(len(x),1024))
    m=(f>0.1)&(f<0.6); return float(f[m][np.argmax(P[m])])
rows=[]
for p in sorted(glob.glob(os.path.join(W,'*_iq.npz'))):
    s=os.path.basename(p)[:-7]
    z=np.load(p); A=z['re'].astype(np.float32)+1j*z['im'].astype(np.float32)
    for r in range(A.shape[0]):
        seg=A[r,:1800]; seg=seg-seg.mean(0,keepdims=True)
        e=bp(np.abs(seg),0.1,0.6).std(0)
        k1=int(np.argmax(e))
        mask=np.ones(len(e),bool); mask[max(k1-6,0):k1+7]=False
        if not mask.any(): continue
        k2=int(np.arange(len(e))[mask][np.argmax(e[mask])])
        a1=bp(np.abs(seg[:,k1]),0.1,0.6); a2=bp(np.abs(seg[:,k2]),0.1,0.6)
        rows.append(dict(s=s,r=r+1,k1=k1+8,k2=k2+8,d=k2-k1,ratio=float(e[k2]/e[k1]),
                         f1=f0(a1),f2=f0(a2)))
import collections
print('전체 %d개 (피험자·레이더)'%len(rows))
rt=np.array([x['ratio'] for x in rows]); dd=np.array([x['d'] for x in rows])
same=np.array([abs(x['f1']-x['f2'])<0.02 for x in rows])
print('2차 봉우리 세기(1차 대비): 중앙값 %.2f, 25~75%% %.2f~%.2f'%(np.median(rt),*np.percentile(rt,[25,75])))
print('2차 봉우리가 1차와 같은 호흡 주파수(±0.02Hz): %d/%d (%.0f%%)'%(same.sum(),len(rows),100*same.mean()))
print('세기 0.3 이상이면서 같은 주파수: %d개'%((rt>=0.3)&same).sum())
print('\n레이더별 2차 봉우리 거리차(bin, 중앙값):')
for r in [1,2,3]:
    d=[x['d'] for x in rows if x['r']==r]; q=[x['ratio'] for x in rows if x['r']==r]
    sm=[abs(x['f1']-x['f2'])<0.02 for x in rows if x['r']==r]
    print('  r%d  Δbin 중앙값 %+d (범위 %+d~%+d) | 세기 %.2f | 같은 주파수 %.0f%%'%(
        r,int(np.median(d)),min(d),max(d),np.median(q),100*np.mean(sm)))
json.dump(rows,open('ghost2.json','w'))
