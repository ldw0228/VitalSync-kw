# -*- coding: utf-8 -*-
"""SNN 후보 선택기 — 9개 후보 파형을 스파이크로 읽고 신뢰도를 매긴다.
   가중치를 공유하는 채점망이 후보를 하나씩 보고 점수를 내고, 그중 최선을 고른다."""
import numpy as np, torch, torch.nn as nn, json, time, sys
import snntorch as snn
from snntorch import surrogate
from enc2 import ENC
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
D=np.load('ds_rate.npz',allow_pickle=True)
X=D['X'].astype(np.float32)            # (N,9,300)
FE=D['feat'].astype(np.float32)        # (N,9,10)
EST=D['est'].astype(np.float32); Y=D['y'].astype(np.float32); ERR=D['err'].astype(np.float32)
SUBJ=D['subj']; BLK=D['blk']
N,C,T=X.shape
DS=3                                    # 300 -> 100 timestep
X=X[:,:,:(T//DS)*DS].reshape(N,C,-1,DS).mean(-1)       # (N,9,100)
T=X.shape[2]
# 후보 간 관계 특징 (라벨 미사용)
med=np.median(EST,axis=1,keepdims=True); dev=np.abs(EST-med)
cons=np.stack([[np.sum(np.abs(EST[i]-EST[i,c])<=1.0)-1 for c in range(C)] for i in range(N)]).astype(np.float32)
AUX=np.concatenate([FE,EST[...,None],dev[...,None],cons[...,None]],axis=2)
mu=AUX.reshape(-1,AUX.shape[2]).mean(0); sd=AUX.reshape(-1,AUX.shape[2]).std(0)+1e-6
AUX=((AUX-mu)/sd).astype(np.float32)
A_DIM=AUX.shape[2]
SUB=sorted(set(SUBJ.tolist())); FOLD=[SUB[i::4] for i in range(4)]

class Scorer(nn.Module):
    """후보 하나를 보고 '이 추정을 얼마나 믿을 수 있나'를 점수로 낸다"""
    def __init__(self,fin,aux,h1=96,h2=48,beta=0.9):
        super().__init__(); sg=surrogate.atan()
        self.fc1=nn.Linear(fin,h1); self.l1=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True)
        self.fc2=nn.Linear(h1,h2);  self.l2=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True)
        self.head=nn.Sequential(nn.Linear(h2+aux,48),nn.ReLU(),nn.Linear(48,1))
    def forward(self,s,a):
        # s: (B*C, T, fin)
        m1=self.l1.init_leaky(); m2=self.l2.init_leaky()
        acc=0.; spk=0.
        for t in range(s.shape[1]):
            c1,m1=self.l1(self.fc1(s[:,t]),m1)
            c2,m2=self.l2(self.fc2(c1),m2)
            acc=acc+c2; spk=spk+c1.mean()+c2.mean()
        emb=acc/s.shape[1]
        return self.head(torch.cat([emb,a],1)).squeeze(1), float(spk.detach())/(2*s.shape[1])

def run(enc,fold,epochs=18,bs=32,lr=2e-3,log=None):
    te=np.isin(SUBJ,FOLD[fold]); tr=~te
    xt=torch.tensor(X[tr]); at=torch.tensor(AUX[tr]); et=torch.tensor(ERR[tr])
    xv=torch.tensor(X[te]); av=torch.tensor(AUX[te]); ev=ERR[te]
    net=Scorer(enc.dim(1),A_DIM)
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=(9e9,None); hist=[]
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(xt))
        for i in range(0,len(perm),bs):
            idx=perm[i:i+bs]; b=len(idx)
            xb=xt[idx].reshape(b*C,T,1); ab=at[idx].reshape(b*C,A_DIM)
            sc,_=net(enc(xb),ab); sc=sc.reshape(b,C)
            tgt=torch.log1p(et[idx])
            # 점수는 오차를 맞히고(회귀), 동시에 최선 후보가 1등이 되도록(순위)
            loss=nn.functional.mse_loss(sc,tgt) \
                 + 0.5*nn.functional.cross_entropy(-sc,et[idx].argmin(1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
        sch.step()
        net.eval(); outs=[]; rates=[]
        with torch.no_grad():
            for i in range(0,len(xv),64):
                b=len(xv[i:i+64])
                sc,rt=net(enc(xv[i:i+64].reshape(b*C,T,1)),av[i:i+64].reshape(b*C,A_DIM))
                outs.append(sc.reshape(b,C)); rates.append(rt)
        sc=torch.cat(outs).numpy(); pick=ev[np.arange(len(ev)),sc.argmin(1)]
        hist.append(dict(epoch=ep+1,mae=float(pick.mean()),p1=float((pick<=1).mean())))
        if log: log('    ep%02d  MAE %.3f  1BPM %.0f%%'%(ep+1,pick.mean(),100*(pick<=1).mean()))
        if pick.mean()<best[0]: best=(float(pick.mean()),pick.copy())
    return dict(fold=fold,best=best[0],pick=best[1].tolist(),hist=hist,
                spike=float(np.mean(rates)),
                params=sum(p.numel() for p in net.parameters()))

if __name__=='__main__':
    which=sys.argv[1] if len(sys.argv)>1 else 'rate'
    enc=[e for e in ENC if e.name==which][0]
    res=[]; t0=time.time()
    for f in range(4):
        r=run(enc,f,log=lambda m:print(m,flush=True))
        res.append(r)
        print('fold%d  MAE %.3f  (%.0f초)'%(f,r['best'],time.time()-t0),flush=True)
    pick=np.concatenate([np.array(r['pick']) for r in res])
    print('\n=== %s ===  MAE %.2f  1BPM %.0f%%  2.5이내 %.0f%%  파라미터 %d'%(
        enc.name,pick.mean(),100*(pick<=1).mean(),100*(pick<=2.5).mean(),res[0]['params']))
    json.dump(res,open('sel_%s.json'%enc.name,'w'),ensure_ascii=False)
