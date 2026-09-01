# -*- coding: utf-8 -*-
"""I/Q 평면 보정 + arctangent 복조 vs 현재 선형 투영
   현재 방식은 원호를 직선에 사영하므로 변조가 크면 sin 비선형이 생긴다(=배음).
   원/타원을 적합해 중심을 잡고 위상을 직접 풀면 그 비선형이 사라진다."""
import numpy as np, json, os
from scipy.signal import butter, filtfilt, welch
W='_work'
SY=json.load(open('sync12_final.json')); FSR=10.0
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def rate(x,fs):
    f,P=welch(x,fs=fs,nperseg=min(len(x),int(51*fs)),nfft=1<<14)
    m=(f>0.06)&(f<0.80); f,P=f[m],P[m]; i=int(np.argmax(P))
    if 0<i<len(P)-1:
        a,b,c=np.log(P[i-1]+1e-30),np.log(P[i]+1e-30),np.log(P[i+1]+1e-30)
        d=np.clip(0.5*(a-c)/(a-2*b+c+1e-30),-1,1); return float((f[i]+d*(f[1]-f[0]))*60)
    return float(f[i]*60)
def circle_fit(x,y):
    """Kåsa 대수적 원 적합"""
    A=np.c_[2*x,2*y,np.ones(len(x))]; b=x**2+y**2
    try: c,*_=np.linalg.lstsq(A,b,rcond=None)
    except Exception: return x.mean(),y.mean()
    return float(c[0]),float(c[1])
def ellipse_fit(x,y):
    """일반 원뿔곡선 적합 -> 타원을 원으로 펴는 변환 반환 (실패 시 None)"""
    D=np.c_[x*x,x*y,y*y,x,y,np.ones(len(x))]
    try:
        _,_,V=np.linalg.svd(D,full_matrices=False); A,B,C,Dd,E,F=V[-1]
    except Exception: return None
    M=np.array([[2*A,B],[B,2*C]])
    if abs(np.linalg.det(M))<1e-18: return None
    if B*B-4*A*C>=0: return None                      # 타원이 아님
    x0,y0=np.linalg.solve(M,[-Dd,-E])
    phi=0.5*np.arctan2(B,A-C)
    ca,sa=np.cos(phi),np.sin(phi)
    Ar=A*ca*ca+B*ca*sa+C*sa*sa; Cr=A*sa*sa-B*ca*sa+C*ca*ca
    Fr=F+A*x0*x0+B*x0*y0+C*y0*y0+Dd*x0+E*y0
    if Ar==0 or Cr==0 or Fr==0: return None
    aa,bb=-Fr/Ar,-Fr/Cr
    if aa<=0 or bb<=0: return None
    return (x0,y0,phi,np.sqrt(aa),np.sqrt(bb))
def phases(c):
    """세 가지 복조 방식의 호흡 파형"""
    x,y=c.real.astype(float),c.imag.astype(float)
    out={}
    # A. 현재 — 선형 주성분 투영
    xy=np.stack([x-x.mean(),y-y.mean()],1)
    _,_,vt=np.linalg.svd(xy,full_matrices=False)
    out['선형 투영(현재)']=xy@vt[0]
    # B. 원 적합 + arctan
    cx,cy=circle_fit(x,y)
    out['원적합+arctan']=np.unwrap(np.arctan2(y-cy,x-cx))
    # C. 타원 적합(=I/Q 불균형 보정) + arctan
    e=ellipse_fit(x,y)
    if e is None: out['타원보정+arctan']=out['원적합+arctan']
    else:
        x0,y0,phi,aa,bb=e
        ca,sa=np.cos(phi),np.sin(phi)
        u=(x-x0)*ca+(y-y0)*sa; v=-(x-x0)*sa+(y-y0)*ca
        v=v*(aa/bb)
        out['타원보정+arctan']=np.unwrap(np.arctan2(v,u))
    return out
BLK=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('회복 호흡',18,46),('운동 후 호흡',172,234)]
BL={(r['s'],r['blk']):r for r in json.load(open('baseline3.json'))}
res={}
for o in SY:
    if o['grade']=='실패': continue
    s=o['s']; apr=o['ap']
    p=os.path.join(W,s+'_iq.npz')
    if not os.path.exists(p): continue
    z=np.load(p); A=z['re'].astype(np.float32)+1j*z['im'].astype(np.float32)
    for nm,a0,a1 in BLK:
        key=(s,nm)
        if key not in BL: continue
        bio=BL[key]['bio']['스펙트럼']
        j0,j1=int((apr+a0)*FSR),int((apr+a1)*FSR)
        if j0<0 or j1>A.shape[1]: continue
        got={}
        for r in range(3):
            sg=A[r,j0:j1].copy(); sg=sg-sg.mean(0,keepdims=True)
            e=bp(np.abs(sg),0.1,0.6,FSR).std(0); k=int(np.argmax(e))
            c=sg[:,max(k-1,0):k+2].mean(1)
            for mk,w in phases(c).items():
                got.setdefault(mk,[]).append(rate(bp(w,0.06,0.80,FSR),FSR))
        for mk,v in got.items():
            res.setdefault(mk,[]).append(dict(s=s,blk=nm,bio=bio,rad=v))
json.dump(res,open('arctan.json','w'),ensure_ascii=False)
print('%-18s %7s %8s %9s %9s'%('복조 방식','MAE','1BPM이내','2.5이내','최선상한'))
for mk,rows in res.items():
    em=np.array([abs(float(np.median(r['rad']))-r['bio']) for r in rows])
    eb=np.array([min(abs(x-r['bio']) for x in r['rad']) for r in rows])
    print('%-18s %7.2f %7.0f%% %8.0f%% %9.2f'%(mk,em.mean(),100*np.mean(em<=1),100*np.mean(em<=2.5),eb.mean()))
print('\n블록별 MAE (중앙값 융합)')
print('%-14s'%'' + ''.join('%18s'%k for k in res))
for nm,_,_ in BLK:
    line='%-14s'%nm
    for mk,rows in res.items():
        e=[abs(float(np.median(r['rad']))-r['bio']) for r in rows if r['blk']==nm]
        line+='%18.2f'%np.mean(e)
    print(line)
