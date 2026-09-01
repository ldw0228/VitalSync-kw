# -*- coding: utf-8 -*-
"""위아래 두 줄이 (a) I/Q 직교쌍인가, (b) 진짜 멀티패스 고스트인가
   구분 기준: 두 파형의 호흡 성분 위상차. I/Q면 ±90° 근처에 몰림. 고스트면 흩어짐."""
import numpy as np, glob, os
from scipy.signal import butter, filtfilt, hilbert
W='_work'
def bp(x,lo=0.1,hi=0.6,fs=10.0,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def phase_diff(a,b):
    """두 실수 파형의 평균 위상차(도). 진폭 가중."""
    za=hilbert(a); zb=hilbert(b)
    w=np.abs(za)*np.abs(zb)
    d=np.angle(za*np.conj(zb))
    c=np.sum(w*np.exp(1j*d))/np.sum(w)
    return np.degrees(np.angle(c)), float(np.abs(c))   # 각도, 집중도(0~1)
IQ=[];GH=[]
for p in sorted(glob.glob(os.path.join(W,'*_iq.npz'))):
    z=np.load(p); re=z['re'].astype(np.float32); im=z['im'].astype(np.float32)
    for r in range(re.shape[0]):
        R=bp(re[r,:1800]-re[r,:1800].mean(0)); I=bp(im[r,:1800]-im[r,:1800].mean(0))
        A=(re[r,:1800]+1j*im[r,:1800]); A=A-A.mean(0)
        e=bp(np.abs(A)).std(0); k=int(np.argmax(e))
        # (a) 같은 bin의 실수 vs 허수  = 선배가 본 "위아래 한 쌍"
        d,c=phase_diff(R[:,k],I[:,k]);  IQ.append((d,c))
        # (b) 올바른 파싱 후 남는 2차 봉우리 (r3의 Δ+30 등)
        m=np.ones(len(e),bool); m[max(k-6,0):k+7]=False
        k2=int(np.arange(len(e))[m][np.argmax(e[m])])
        pa=bp(np.abs(A[:,k])); pb=bp(np.abs(A[:,k2]))
        d2,c2=phase_diff(pa,pb); GH.append((d2,c2,k2-k,r+1))
IQ=np.array(IQ)
print('■ 같은 bin의 실수 vs 허수  (n=%d)'%len(IQ))
ang=IQ[:,0]; absang=np.abs(ang)
print('  위상차 |각도| 중앙값 %.0f°   (25~75%%: %.0f~%.0f°)'%(np.median(absang),*np.percentile(absang,[25,75])))
print('  70~110° 안에 드는 비율: %.0f%%   위상 집중도 중앙값 %.2f'%(
    100*np.mean((absang>70)&(absang<110)), np.median(IQ[:,1])))
G=np.array([[d,c] for d,c,dk,r in GH]); DK=np.array([dk for *_ ,dk,r in [(g) for g in GH]])
print('\n■ 올바른 파싱 후 남는 2차 봉우리  (n=%d)'%len(GH))
ga=np.abs(G[:,0])
print('  위상차 |각도| 중앙값 %.0f°   (25~75%%: %.0f~%.0f°)'%(np.median(ga),*np.percentile(ga,[25,75])))
print('  70~110° 안에 드는 비율: %.0f%%'%(100*np.mean((ga>70)&(ga<110))))
for r in [1,2,3]:
    sub=[abs(d) for d,c,dk,rr in GH if rr==r]
    print('    r%d 위상차 중앙값 %.0f° (n=%d)'%(r,np.median(sub),len(sub)))
