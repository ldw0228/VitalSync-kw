# -*- coding: utf-8 -*-
"""두 자르기 방식으로 각각 데이터셋 생성 (후보 9개 방식, 동일 조건)"""
import numpy as np, json, os
from scipy.signal import butter, filtfilt, welch
W='_work'
SY={o['s']:o for o in json.load(open('sync12_final.json'))}
XL=json.load(open('xl_exp2.json'))
FSR=10.0; WIN=30.0; HOP=5.0; T=int(WIN*FSR); NB=8; SCRIPT=135.0
BLK=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('회복 호흡',16,78),('운동 후 호흡',172,234)]
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def spec(x,fs):
    f,P=welch(x,fs=fs,nperseg=min(len(x),int(WIN*fs)),nfft=1<<14)
    m=(f>0.06)&(f<0.80); return f[m],P[m]
def rate_feat(x,fs):
    f,P=spec(x,fs); i=int(np.argmax(P))
    if 0<i<len(P)-1:
        a,b,c=np.log(P[i-1]+1e-30),np.log(P[i]+1e-30),np.log(P[i+1]+1e-30)
        d=np.clip(0.5*(a-c)/(a-2*b+c+1e-30),-1,1); f0=f[i]+d*(f[1]-f[0])
    else: f0=f[i]
    tot=P.sum()+1e-30
    band=lambda fc,bw=0.03: float(P[(f>fc-bw)&(f<fc+bw)].sum())
    p1=band(f0); Pn=P/tot; srt=np.sort(P)[::-1]; h=len(x)//2
    q=lambda seg:(lambda ff,PP: float(ff[np.argmax(PP)]))(*spec(seg,fs))
    return float(f0*60), np.array([band(2*f0)/(p1+1e-30),band(3*f0)/(p1+1e-30),p1/tot,
        float(-(Pn*np.log(Pn+1e-30)).sum()/np.log(len(Pn))),
        float(srt[1]/(srt[0]+1e-30)),abs(q(x[:h])-q(x[h:]))*60,float(np.std(x))],np.float32)
def circle(x,y):
    A=np.c_[2*x,2*y,np.ones(len(x))]; b=x*x+y*y
    try: c,*_=np.linalg.lstsq(A,b,rcond=None)
    except Exception: return x.mean(),y.mean(),0.
    cx,cy=float(c[0]),float(c[1]); r=np.sqrt(max(c[2]+cx*cx+cy*cy,1e-12))
    return cx,cy,float(np.std(np.hypot(x-cx,y-cy)-r)/(r+1e-12))
def ellipse(x,y):
    D=np.c_[x*x,x*y,y*y,x,y,np.ones(len(x))]
    try: _,_,Vt=np.linalg.svd(D,full_matrices=False); A_,B_,C_,Dd,E_,F_=Vt[-1]
    except Exception: return None
    M=np.array([[2*A_,B_],[B_,2*C_]])
    if abs(np.linalg.det(M))<1e-18 or B_*B_-4*A_*C_>=0: return None
    x0,y0=np.linalg.solve(M,[-Dd,-E_]); phi=0.5*np.arctan2(B_,A_-C_)
    ca,sa=np.cos(phi),np.sin(phi)
    Ar=A_*ca*ca+B_*ca*sa+C_*sa*sa; Cr=A_*sa*sa-B_*ca*sa+C_*ca*ca
    Fr=F_+A_*x0*x0+B_*x0*y0+C_*y0*y0+Dd*x0+E_*y0
    if Ar==0 or Cr==0 or Fr==0: return None
    aa,bb=-Fr/Ar,-Fr/Cr
    if aa<=0 or bb<=0: return None
    return x0,y0,phi,float(np.sqrt(aa)),float(np.sqrt(bb))
def demods(c):
    x,y=c.real.astype(float),c.imag.astype(float); out=[];ex=[]
    xy=np.stack([x-x.mean(),y-y.mean()],1); _,_,vt=np.linalg.svd(xy,full_matrices=False)
    out.append(xy@vt[0]); ex.append(0.)
    cx,cy,res=circle(x,y); out.append(np.unwrap(np.arctan2(y-cy,x-cx))); ex.append(res)
    e=ellipse(x,y)
    if e is None: out.append(out[-1].copy()); ex.append(res)
    else:
        x0,y0,phi,aa,bb=e; ca,sa=np.cos(phi),np.sin(phi)
        u=(x-x0)*ca+(y-y0)*sa; v=(-(x-x0)*sa+(y-y0)*ca)*(aa/bb)
        out.append(np.unwrap(np.arctan2(v,u))); ex.append(float(abs(np.log(aa/bb))))
    return out,ex
def start(v):
    if not v or '?' in str(v): return None
    try: return float(str(v).split('~')[0])
    except Exception: return None
def build(mode):
    X=[];EST=[];FE=[];Y=[];SU=[];BLn=[];TT=[]
    for s,o in sorted(SY.items()):
        num=int(s[1:3])
        pm=os.path.join(W,s+'_mot.npz'); pi=os.path.join(W,s+'_iq.npz')
        if not(os.path.exists(pm) and os.path.exists(pi)): continue
        if mode=='marker':
            xr=start(XL.get(str(num),{}).get('rad')); xb_=start(XL.get(str(num),{}).get('bio'))
            if xr is None or xb_ is None or s=='S01_CMS': continue
            apr,apb=xr+SCRIPT,xb_+SCRIPT
        else:
            if o['grade']=='실패': continue
            apr,apb=o['ap'],o['ap']+o['off']
        z=np.load(pm); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
        xb=bp(rsp,0.06,0.80,fsb)
        zi=np.load(pi); A=zi['re'].astype(np.float32)+1j*zi['im'].astype(np.float32)
        for nm,a0,a1 in BLK:
            j0b,j1b=int((apr+a0)*FSR),int((apr+a1)*FSR)
            if j0b<0 or j1b>A.shape[1]: continue
            kb=[]
            for r in range(3):
                sg=A[r,j0b:j1b]-A[r,j0b:j1b].mean(0,keepdims=True)
                kb.append(int(np.argmax(bp(np.abs(sg),0.1,0.6,FSR).std(0))))
            t=a0
            while t+WIN<=a1:
                jb0,jb1=int((apb+t)*fsb),int((apb+t+WIN)*fsb)
                jr0,jr1=int((apr+t)*FSR),int((apr+t+WIN)*FSR)
                if jb0<0 or jb1>len(rsp) or jr1>A.shape[1]: t+=HOP; continue
                yb,_=rate_feat(xb[jb0:jb1],fsb)
                ws=[];es=[];fs_=[]
                for r in range(3):
                    sg=A[r,jr0:jr1].copy(); sg=sg-sg.mean(0,keepdims=True)
                    k=kb[r]; c=sg[:,max(k-1,0):k+2].mean(1)
                    ds,ex=demods(c)
                    for di,w in enumerate(ds):
                        wv=bp(w,0.06,0.80,FSR)
                        if len(wv)<T: wv=np.pad(wv,(0,T-len(wv)),mode='edge')
                        wv=wv[:T]; est,ft=rate_feat(wv,FSR)
                        ws.append((wv/(np.std(wv)+1e-12)).astype(np.float32)); es.append(est)
                        fs_.append(np.concatenate([ft,[ex[di],float(r),float(di)]]).astype(np.float32))
                X.append(np.stack(ws)); EST.append(es); FE.append(np.stack(fs_))
                Y.append(yb); SU.append(s); BLn.append(nm); TT.append(t)
                t+=HOP
    X=np.stack(X); EST=np.array(EST,np.float32); FE=np.stack(FE); Y=np.array(Y,np.float32)
    np.savez_compressed('ds_%s.npz'%mode,X=X,est=EST,feat=FE,y=Y,err=np.abs(EST-Y[:,None]),
                        subj=np.array(SU),blk=np.array(BLn),t=np.array(TT,np.float32))
    e=np.abs(np.median(EST[:,[0,3,6]],1)-Y)
    print('%s: 창 %d · 피험자 %d · 고전 MAE %.2f (1BPM %.0f%%)'%(
        mode,len(Y),len(set(SU)),e.mean(),100*np.mean(e<=1)),flush=True)
for m in ['marker','landmark']: build(m)
