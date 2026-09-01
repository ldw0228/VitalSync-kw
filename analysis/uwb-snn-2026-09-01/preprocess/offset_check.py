# -*- coding: utf-8 -*-
"""184값 평면에서 아래 줄(실수 블록)과 위 줄(허수 블록)의 봉우리 간격이 일정한가"""
import numpy as np, glob, os
from scipy.signal import butter, filtfilt
W='_work'
def bp(x,lo=0.1,hi=0.6,fs=10.0,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
d=[]
for p in sorted(glob.glob(os.path.join(W,'*_iq.npz'))):
    z=np.load(p); re=z['re'].astype(np.float32); im=z['im'].astype(np.float32)
    NB=re.shape[2]
    for r in range(re.shape[0]):
        R=bp(re[r,:1800]-re[r,:1800].mean(0)).std(0)
        I=bp(im[r,:1800]-im[r,:1800].mean(0)).std(0)
        d.append(int(np.argmax(I))-int(np.argmax(R)))
d=np.array(d)
print('저장 블록 크기 = %d bin (원본으로는 92 bin)'%NB)
print('아래줄 봉우리 → 위줄 봉우리 간격 (블록 크기를 뺀 값):')
print('  0 인 경우: %d/%d (%.0f%%)   |  ±1 이내: %.0f%%   |  ±3 이내: %.0f%%'%(
    (d==0).sum(),len(d),100*np.mean(d==0),100*np.mean(np.abs(d)<=1),100*np.mean(np.abs(d)<=3)))
print('  표준편차 %.2f bin,  범위 %+d ~ %+d'%(d.std(),d.min(),d.max()))
