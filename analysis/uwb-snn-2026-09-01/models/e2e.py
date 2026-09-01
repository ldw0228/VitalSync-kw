# -*- coding: utf-8 -*-
"""1단계 — 레이더 신호를 그대로 넣어 호흡수를 직접 예측
   입력: 가슴 주변 17 range bin × (I,Q) × 레이더 3대 = 102 채널
   전처리: 되먹임 필터로 정적 배경 제거 (빈 방 파일 불필요)
   출력: 호흡수 1개.  비교 대상: 고전 방식 1.66 BPM"""
import numpy as np, json, os, sys, time
from scipy.signal import butter, filtfilt
W='_work'
V=''
FSR=10.0; WIN=30.0; HOP=5.0; T=int(WIN*FSR); NB=8
BLK=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('회복 호흡',16,78),('운동 후 호흡',172,234)]
BETA=0.98            # 되먹임 필터: 정적 배경 추정 (차단 ~0.03 Hz)

def loopback(x, beta=BETA):
    """c_n = b*c_{n-1} + (1-b)*r_n  ;  출력 = r_n - c_n"""
    c=np.zeros_like(x); acc=x[0].copy()
    for n in range(len(x)):
        acc=beta*acc+(1-beta)*x[n]; c[n]=acc
    return x-c

def build():
    SY=json.load(open(V+'sync12_final.json'))
    D0=np.load(V+'ds_rate.npz',allow_pickle=True)     # 정답·메타 재사용
    key={(s,b,round(float(t),1)):i for i,(s,b,t) in enumerate(zip(D0['subj'],D0['blk'],D0['t']))}
    X=np.zeros((len(D0['y']),T,102),dtype=np.float32); ok=np.zeros(len(D0['y']),bool)
    def bp(x,lo,hi,fs,o=3):
        b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
    for o in SY:
        if o['grade']=='실패': continue
        s=o['s']; apr=o['ap']
        p=os.path.join(W,s+'_iq.npz')
        if not os.path.exists(p): continue
        z=np.load(p); A=z['re'].astype(np.float32)+1j*z['im'].astype(np.float32)
        for nm,a0,a1 in BLK:
            j0b,j1b=int((apr+a0)*FSR),int((apr+a1)*FSR)
            if j0b<0 or j1b>A.shape[1]: continue
            seg=[]
            for r in range(3):
                raw=A[r,j0b:j1b]
                cl=loopback(raw)                       # 배경 제거
                e=bp(np.abs(cl),0.1,0.6,FSR).std(0); k=int(np.argmax(e))
                b0=max(k-NB,0); b1=min(k+NB+1,A.shape[2])
                if b1-b0<2*NB+1: b0=max(min(b0,A.shape[2]-(2*NB+1)),0); b1=b0+2*NB+1
                sub=cl[:,b0:b1]
                sc=np.abs(sub).std()+1e-9              # 레이더별 크기 정규화
                seg.append(np.concatenate([sub.real,sub.imag],1)/sc)
            seg=np.concatenate(seg,1)                  # (L,102)
            t=a0
            while t+WIN<=a1:
                i=key.get((s,nm,round(t,1)))
                if i is not None:
                    j=int((t-a0)*FSR)
                    w=seg[j:j+T]
                    if len(w)==T: X[i]=w; ok[i]=True
                t+=HOP
    np.savez_compressed(V+'ds_e2e.npz',X=X,ok=ok,y=D0['y'],subj=D0['subj'],blk=D0['blk'])
    print('입력',X.shape,'유효',int(ok.sum()),'/',len(ok),flush=True)

if sys.argv[1]=='build': build(); sys.exit()

import torch, torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from enc2 import ENC
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
D=np.load(V+'ds_e2e.npz',allow_pickle=True)
X=D['X'][D['ok']]; Y=D['y'][D['ok']].astype(np.float32)
SUBJ=D['subj'][D['ok']]; BLKn=D['blk'][D['ok']]
DS=3; N,Tn,F=X.shape
X=X[:,:(Tn//DS)*DS].reshape(N,-1,DS,F).mean(2).astype(np.float32)
Tn=X.shape[1]
ym,ys=Y.mean(),Y.std()
SUB=sorted(set(SUBJ.tolist())); FOLD=[SUB[i::4] for i in range(4)]
class Net(nn.Module):
    def __init__(self,fin,h1=128,h2=64,beta=0.9):
        super().__init__(); sg=surrogate.atan()
        self.fc1=nn.Linear(fin,h1); self.l1=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True)
        self.fc2=nn.Linear(h1,h2);  self.l2=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True)
        self.out=nn.Linear(h2,1)
    def forward(self,s):
        m1=self.l1.init_leaky(); m2=self.l2.init_leaky(); acc=0.; spk=0.
        for t in range(s.shape[1]):
            c1,m1=self.l1(self.fc1(s[:,t]),m1)
            c2,m2=self.l2(self.fc2(c1),m2)
            acc=acc+c2; spk=spk+c1.mean()+c2.mean()
        return self.out(acc/s.shape[1]).squeeze(1), float(spk.detach())/(2*s.shape[1])
def run(enc,fold,epochs=30,bs=32,lr=3e-3):
    te=np.isin(SUBJ,FOLD[fold]); tr=~te
    xt=torch.tensor(X[tr]); yt=torch.tensor((Y[tr]-ym)/ys)
    xv=torch.tensor(X[te]); yv=Y[te]
    net=Net(enc.dim(F)); opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=(9e9,None,0.)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(xt))
        for i in range(0,len(perm),bs):
            idx=perm[i:i+bs]
            p,_=net(enc(xt[idx])); loss=nn.functional.smooth_l1_loss(p,yt[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
        sch.step(); net.eval(); ps=[]; rs=[]
        with torch.no_grad():
            for i in range(0,len(xv),64):
                p,r=net(enc(xv[i:i+64])); ps.append(p); rs.append(r)
        pr=torch.cat(ps).numpy()*ys+ym; e=np.abs(pr-yv)
        if e.mean()<best[0]: best=(float(e.mean()),e.copy(),float(np.mean(rs)))
    return best,sum(p.numel() for p in net.parameters())
which=sys.argv[1]
enc=[e for e in ENC if e.name==which][0]
errs=[]; t0=time.time()
for f in range(4):
    b,pn=run(enc,f); errs.append(b[1])
    print('fold%d MAE %.2f'%(f,b[0]),flush=True)
e=np.concatenate(errs)
print('\n=== %s ===  MAE %.2f  1BPM %.0f%%  2.5이내 %.0f%%  파라미터 %d  %.0f초'%(
    enc.name,e.mean(),100*(e<=1).mean(),100*(e<=2.5).mean(),pn,time.time()-t0))
json.dump(dict(enc=which,mae=float(e.mean()),p1=float((e<=1).mean()),
               p25=float((e<=2.5).mean()),params=pn,err=e.tolist()),
          open(V+'e2e_%s.json'%which,'w'),ensure_ascii=False)
