# -*- coding: utf-8 -*-
"""과제 A(자세 각도) 데이터셋: 레이더 1대 입력, |각도| ∈ {0,45,90}"""
import numpy as np, os, glob, json
from scipy.signal import butter, filtfilt
W='_work'
LM=json.load(open('landmarks.json'))
FSR=10.0; DS=2; FS=FSR/DS      # 5 Hz
WIN_S=20.0; T=int(WIN_S*FS)    # 100 timestep
NBIN=8                          # 가슴 ±8 bin  -> 17 bin
ABS=np.array([[0,45,90],[45,0,45],[90,45,0]])
CLS={0:0,45:1,90:2}
TARGET={0:12,45:9,90:18}        # 클래스별 (피험자·레이더·블록)당 목표 윈도우 수 -> 균형

def bp(x,lo,hi,fs=FSR,order=3):
    b,a=butter(order,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)

X=[];y=[];g=[];rad=[];blk=[];ang=[]
for p in sorted(glob.glob(os.path.join(W,'*_iq.npz'))):
    subj=os.path.basename(p)[:-7]
    if subj not in LM: continue
    z=np.load(p); A=z['re'].astype(np.float32)+1j*z['im'].astype(np.float32)
    bl=LM[subj]['blocks']
    lo=max(int(bl[0][0]*FSR),0); hi=min(int(bl[2][1]*FSR),A.shape[1])
    for r in range(A.shape[0]):
        seg_all=A[r,lo:hi]
        seg_all=seg_all-seg_all.mean(0,keepdims=True)
        e=bp(np.abs(seg_all),0.1,0.6).std(axis=0)
        kb=int(np.argmax(e))
        b0,b1=max(kb-NBIN,0), min(kb+NBIN+1, A.shape[2])
        if b1-b0 < 2*NBIN+1:
            b0=max(min(b0, A.shape[2]-(2*NBIN+1)),0); b1=b0+2*NBIN+1
        # bin별 주성분 방향을 1번 전체에서 고정
        sub=seg_all[:,b0:b1]
        dirs=[]
        for k in range(sub.shape[1]):
            xy=np.stack([sub[:,k].real,sub[:,k].imag],1); xy=xy-xy.mean(0)
            _,_,vt=np.linalg.svd(xy,full_matrices=False); dirs.append(vt[0])
        dirs=np.array(dirs)                                   # (nb,2)
        # 세션 보정 스케일: 1번 전체 호흡 진폭
        proj_all=np.stack([ (np.stack([sub[:,k].real,sub[:,k].imag],1)@dirs[k]) for k in range(sub.shape[1])],1)
        proj_all=bp(proj_all,0.1,0.6)
        scale=float(proj_all.std())+1e-9
        for bi,(a,bb) in enumerate(bl):
            j0,j1=int(a*FSR),int(bb*FSR)
            if j0<lo or j1>hi or j1-j0<int(WIN_S*FSR)+20: continue
            s=A[r,j0:j1].copy(); s=s-s.mean(0,keepdims=True); s=s[:,b0:b1]
            proj=np.stack([(np.stack([s[:,k].real,s[:,k].imag],1)@dirs[k]) for k in range(s.shape[1])],1)
            proj=bp(proj,0.08,0.8)/scale
            mag =bp(np.abs(s),0.08,0.8)/scale
            F=np.concatenate([proj,mag],axis=1)               # (L, 2*nb)
            F=F[:(len(F)//DS)*DS].reshape(-1,DS,F.shape[1]).mean(1)   # 5 Hz
            aang=int(ABS[r,bi]); span=len(F)-T
            if span<=0: continue
            n=TARGET[aang]; stride=max(1, span//max(n-1,1))
            for st in range(0,span+1,stride):
                X.append(F[st:st+T].astype(np.float16)); y.append(CLS[aang])
                g.append(subj); rad.append(r+1); blk.append(bi+1); ang.append(aang)
X=np.stack(X); y=np.array(y); g=np.array(g); rad=np.array(rad); blk=np.array(blk); ang=np.array(ang)
print('X',X.shape,'| 클래스 분포',{k:int((y==v).sum()) for k,v in CLS.items()})
print('피험자',len(set(g.tolist())),'| 레이더별',{r:int((rad==r).sum()) for r in [1,2,3]})
np.savez_compressed('dataset_A.npz',X=X,y=y,g=g,rad=rad,blk=blk,ang=ang,
                    classes=np.array([0,45,90]))
