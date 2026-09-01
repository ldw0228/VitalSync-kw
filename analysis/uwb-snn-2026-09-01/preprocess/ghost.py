# -*- coding: utf-8 -*-
"""고스트/라인오브사이트 주장 검증:
   (a) 184값을 거리 bin으로 그리면 위아래 두 줄이 보이는가  (b) 올바른 파싱 후에도 남는가"""
import numpy as np, glob, os
from scipy.signal import butter, filtfilt
RAW=None
W='_work'
FL=185
def bp(x,lo,hi,fs=10.0,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def peaks(e,k=3):
    """상위 봉우리 bin과 값"""
    idx=np.argsort(e)[::-1]; out=[]
    for i in idx:
        if all(abs(i-j)>4 for j,_ in out): out.append((int(i),float(e[i])))
        if len(out)==k: break
    return out
for p in sorted(glob.glob(os.path.join(W,'*_iq.npz')))[:6]:
    s=os.path.basename(p)[:-7]
    z=np.load(p); re=z['re'].astype(np.float32); im=z['im'].astype(np.float32)
    print('==',s,' re',re.shape)
    for r in range(re.shape[0]):
        seg_c=(re[r]+1j*im[r])[:2000]              # 올바른 파싱 (복소 92 bin)
        seg_c=seg_c-seg_c.mean(0,keepdims=True)
        e_c=bp(np.abs(seg_c),0.1,0.6).std(0)
        # 184값을 모두 실수 거리 bin으로 간주 (선배가 본 화면 재현)
        flat=np.concatenate([re[r][:2000],im[r][:2000]],axis=1)
        flat=flat-flat.mean(0,keepdims=True)
        e_f=bp(flat,0.1,0.6).std(0)
        pc=peaks(e_c,3); pf=peaks(e_f,3)
        print('  r%d  복소92: %s'%(r+1,' '.join('bin%d(%.2f)'%(b,v/pc[0][1]) for b,v in pc)),
              '| 184평면: %s'%' '.join('bin%d(%.2f)'%(b,v/pf[0][1]) for b,v in pf))
