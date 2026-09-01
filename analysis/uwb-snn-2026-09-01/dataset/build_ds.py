# -*- coding: utf-8 -*-
"""호흡수 회귀·후보 선택 데이터셋
   창: 30초 / 5초 이동 · 구간: 평소·느린·회복·운동후 (각 62초)
   후보: 레이더 3대 × 복조 3방식 = 9개"""
import numpy as np, json, os
from scipy.signal import butter, filtfilt, welch
W='_work'
SY=json.load(open('sync12_final.json')); FSR=10.0
WIN=30.0; HOP=5.0; T=int(WIN*FSR)
BLK=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('회복 호흡',16,78),('운동 후 호흡',172,234)]
DEMOD=['선형','원+arctan','타원+arctan']
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def spec(x,fs):
    f,P=welch(x,fs=fs,nperseg=min(len(x),int(WIN*fs)),nfft=1<<14)
    m=(f>0.06)&(f<0.80); return f[m],P[m]
def rate_feat(x,fs):
    """호흡수 + 그 추정을 얼마나 믿을 만한지 말해주는 특징들"""
    f,P=spec(x,fs); i=int(np.argmax(P))
    if 0<i<len(P)-1:
        a,b,c=np.log(P[i-1]+1e-30),np.log(P[i]+1e-30),np.log(P[i+1]+1e-30)
        d=np.clip(0.5*(a-c)/(a-2*b+c+1e-30),-1,1); f0=f[i]+d*(f[1]-f[0])
    else: f0=f[i]
    tot=P.sum()+1e-30
    def band(fc,bw=0.03):
        m=(f>fc-bw)&(f<fc+bw); return float(P[m].sum()) if m.any() else 0.0
    p1=band(f0)
    h2=band(2*f0)/(p1+1e-30); h3=band(3*f0)/(p1+1e-30)
    conc=p1/tot                                   # 기본파가 전체에서 차지하는 비중
    Pn=P/tot; ent=float(-(Pn*np.log(Pn+1e-30)).sum()/np.log(len(Pn)))  # 스펙트럼 엔트로피
    srt=np.sort(P)[::-1]
    pk2=float(srt[1]/(srt[0]+1e-30)) if len(srt)>1 else 0.0             # 2등/1등 봉우리 비
    # 시간축 안정성: 앞뒤 절반의 추정 차이
    h=len(x)//2
    def q(seg):
        ff,PP=spec(seg,fs); return float(ff[np.argmax(PP)])
    drift=abs(q(x[:h])-q(x[h:]))*60
    return float(f0*60), np.array([h2,h3,conc,ent,pk2,drift,float(np.std(x))],dtype=np.float32)
def circle_fit(x,y):
    A=np.c_[2*x,2*y,np.ones(len(x))]; b=x*x+y*y
    try: c,*_=np.linalg.lstsq(A,b,rcond=None)
    except Exception: return x.mean(),y.mean(),0.0
    cx,cy=float(c[0]),float(c[1])
    r=np.sqrt(np.maximum(c[2]+cx*cx+cy*cy,1e-12))
    res=float(np.std(np.hypot(x-cx,y-cy)-r)/(r+1e-12))
    return cx,cy,res
def ellipse_fit(x,y):
    D=np.c_[x*x,x*y,y*y,x,y,np.ones(len(x))]
    try:
        _,_,V=np.linalg.svd(D,full_matrices=False); A,B,C,Dd,E,F=V[-1]
    except Exception: return None
    M=np.array([[2*A,B],[B,2*C]])
    if abs(np.linalg.det(M))<1e-18 or B*B-4*A*C>=0: return None
    x0,y0=np.linalg.solve(M,[-Dd,-E]); phi=0.5*np.arctan2(B,A-C)
    ca,sa=np.cos(phi),np.sin(phi)
    Ar=A*ca*ca+B*ca*sa+C*sa*sa; Cr=A*sa*sa-B*ca*sa+C*ca*ca
    Fr=F+A*x0*x0+B*x0*y0+C*y0*y0+Dd*x0+E*y0
    if Ar==0 or Cr==0 or Fr==0: return None
    aa,bb=-Fr/Ar,-Fr/Cr
    if aa<=0 or bb<=0: return None
    return x0,y0,phi,float(np.sqrt(aa)),float(np.sqrt(bb))
def demods(c):
    x,y=c.real.astype(float),c.imag.astype(float)
    out=[]; extra=[]
    xy=np.stack([x-x.mean(),y-y.mean()],1)
    _,_,vt=np.linalg.svd(xy,full_matrices=False)
    out.append(xy@vt[0]); extra.append(0.0)
    cx,cy,res=circle_fit(x,y)
    out.append(np.unwrap(np.arctan2(y-cy,x-cx))); extra.append(res)
    e=ellipse_fit(x,y)
    if e is None: out.append(out[-1].copy()); extra.append(res)
    else:
        x0,y0,phi,aa,bb=e; ca,sa=np.cos(phi),np.sin(phi)
        u=(x-x0)*ca+(y-y0)*sa; v=(-(x-x0)*sa+(y-y0)*ca)*(aa/bb)
        out.append(np.unwrap(np.arctan2(v,u))); extra.append(float(abs(np.log(aa/bb))))
    return out,extra

X=[];EST=[];FEA=[];Y=[];SUBJ=[];BLKN=[];TSTART=[]
for o in SY:
    if o['grade']=='실패': continue
    s=o['s']; apr=o['ap']; apb=apr+o['off']
    pm=os.path.join(W,s+'_mot.npz'); pi=os.path.join(W,s+'_iq.npz')
    if not(os.path.exists(pm) and os.path.exists(pi)): continue
    z=np.load(pm); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    zi=np.load(pi); A=zi['re'].astype(np.float32)+1j*zi['im'].astype(np.float32)
    for bi,(nm,a0,a1) in enumerate(BLK):
        # 구간 안에서 가슴 bin을 레이더별로 고정
        j0b,j1b=int((apr+a0)*FSR),int((apr+a1)*FSR)
        if j0b<0 or j1b>A.shape[1]: continue
        kb=[]
        for r in range(3):
            sg=A[r,j0b:j1b]-A[r,j0b:j1b].mean(0,keepdims=True)
            e=bp(np.abs(sg),0.1,0.6,FSR).std(0); kb.append(int(np.argmax(e)))
        t=a0
        while t+WIN<=a1:
            jb0,jb1=int((apb+t)*fsb),int((apb+t+WIN)*fsb)
            jr0,jr1=int((apr+t)*FSR),int((apr+t+WIN)*FSR)
            if jb0<0 or jb1>len(rsp) or jr1>A.shape[1]: t+=HOP; continue
            ybpm,_=rate_feat(bp(rsp[jb0:jb1],0.06,0.80,fsb),fsb)
            ws=[];es=[];fs_=[]
            for r in range(3):
                sg=A[r,jr0:jr1].copy(); sg=sg-sg.mean(0,keepdims=True)
                k=kb[r]; c=sg[:,max(k-1,0):k+2].mean(1)
                ds,ex=demods(c)
                for di,w in enumerate(ds):
                    wb=bp(w,0.06,0.80,FSR)
                    if len(wb)<T: wb=np.pad(wb,(0,T-len(wb)),mode='edge')
                    wb=wb[:T]
                    est,ft=rate_feat(wb,FSR)
                    sd=np.std(wb)+1e-12
                    ws.append((wb/sd).astype(np.float32)); es.append(est)
                    fs_.append(np.concatenate([ft,[ex[di],float(r),float(di)]]).astype(np.float32))
            X.append(np.stack(ws)); EST.append(es); FEA.append(np.stack(fs_))
            Y.append(ybpm); SUBJ.append(s); BLKN.append(nm); TSTART.append(t)
            t+=HOP
X=np.stack(X).astype(np.float32); EST=np.array(EST,dtype=np.float32)
FEA=np.stack(FEA).astype(np.float32); Y=np.array(Y,dtype=np.float32)
SUBJ=np.array(SUBJ); BLKN=np.array(BLKN); TSTART=np.array(TSTART,dtype=np.float32)
ERR=np.abs(EST-Y[:,None])
np.savez_compressed('ds_rate.npz',X=X,est=EST,feat=FEA,y=Y,err=ERR,
                    subj=SUBJ,blk=BLKN,t=TSTART,
                    cand=np.array([f'r{r+1}·{d}' for r in range(3) for d in DEMOD]),
                    featname=np.array(['h2','h3','기본파비중','스펙트럼엔트로피','2등봉우리비','전후드리프트','표준편차','적합잔차','레이더','복조']))
print('X',X.shape,'| 창 %d개 · 피험자 %d명'%(len(Y),len(set(SUBJ.tolist()))))
import collections
print('구간별',dict(collections.Counter(BLKN.tolist())))
print('정답 호흡수 %.1f ~ %.1f (중앙값 %.1f)'%(Y.min(),Y.max(),np.median(Y)))
med=np.median(EST,axis=1)
print('\n현재 방식(9후보 중앙값) MAE %.2f'%np.mean(np.abs(med-Y)))
med3=np.median(EST[:,[0,3,6]],axis=1)
print('선형 3대 중앙값        MAE %.2f'%np.mean(np.abs(med3-Y)))
print('9후보 최선(상한)       MAE %.2f'%ERR.min(1).mean())
print('후보별 단독 MAE:',' '.join('%s %.2f'%(c,ERR[:,i].mean()) for i,c in enumerate(
    [f'r{r+1}{d}' for r in range(3) for d in ['선','원','타']])))
