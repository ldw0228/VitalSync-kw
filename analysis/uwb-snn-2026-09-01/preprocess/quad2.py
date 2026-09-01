# -*- coding: utf-8 -*-
import numpy as np, glob, os
from scipy.signal import butter, filtfilt
W='_work'
def bp(x,lo=0.1,hi=0.6,fs=10.0,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def cc(a,b):
    a=a-a.mean(); b=b-b.mean()
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
RI=[];GG=[]
for p in sorted(glob.glob(os.path.join(W,'*_iq.npz'))):
    z=np.load(p); re=z['re'].astype(np.float32); im=z['im'].astype(np.float32)
    for r in range(re.shape[0]):
        R=bp(re[r,:1800]-re[r,:1800].mean(0)); I=bp(im[r,:1800]-im[r,:1800].mean(0))
        A=(re[r,:1800]+1j*im[r,:1800]); A=A-A.mean(0)
        e=bp(np.abs(A)).std(0); k=int(np.argmax(e))
        RI.append(cc(R[:,k],I[:,k]))
        m=np.ones(len(e),bool); m[max(k-6,0):k+7]=False
        k2=int(np.arange(len(e))[m][np.argmax(e[m])])
        GG.append((cc(bp(np.abs(A))[:,k],bp(np.abs(A))[:,k2]),k2-k,r+1))
RI=np.abs(np.array(RI))
print('■ 같은 bin  실수 vs 허수  파형 상관 |r|  (n=%d)'%len(RI))
print('   중앙값 %.2f  |  0.8 이상 %.0f%%  |  0.5 이상 %.0f%%'%(
    np.median(RI),100*np.mean(RI>0.8),100*np.mean(RI>0.5)))
g=np.abs(np.array([x[0] for x in GG]))
print('\n■ 올바른 파싱 후 2차 봉우리 vs 1차  파형 상관 |r|')
print('   중앙값 %.2f  |  0.8 이상 %.0f%%  |  0.5 이상 %.0f%%'%(
    np.median(g),100*np.mean(g>0.8),100*np.mean(g>0.5)))
for r in [1,2,3]:
    s=[abs(x[0]) for x in GG if x[2]==r]; d=[x[1] for x in GG if x[2]==r]
    print('     r%d  |r| 중앙값 %.2f  Δbin %+d (범위 %+d~%+d)'%(r,np.median(s),int(np.median(d)),min(d),max(d)))
