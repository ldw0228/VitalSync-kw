# -*- coding: utf-8 -*-
"""후보를 하나씩 보지 말고 9개를 한꺼번에 보게 하면, 합의(consensus)를 스스로 배울 수 있는가.
   손으로 만든 특징 없이 파형만 준다."""
import numpy as np, torch, torch.nn as nn, json, time
import snntorch as snn
from snntorch import surrogate
from enc2 import ENC
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
D=np.load('ds_rate.npz',allow_pickle=True)
X=D['X'].astype(np.float32); EST=D['est'].astype(np.float32)
Y=D['y'].astype(np.float32); ERR=D['err'].astype(np.float32); SUBJ=D['subj']
N,C,T=X.shape; DS=3
X=X[:,:,:(T//DS)*DS].reshape(N,C,-1,DS).mean(-1); T=X.shape[2]
Xt=np.transpose(X,(0,2,1))                      # (N,T,9) — 9개 후보를 채널로
SUB=sorted(set(SUBJ.tolist())); FOLD=[SUB[i::4] for i in range(4)]
enc=[e for e in ENC if e.name=='rate'][0]
class Joint(nn.Module):
    def __init__(self,fin,h1=128,h2=64,beta=0.9):
        super().__init__(); sg=surrogate.atan()
        self.fc1=nn.Linear(fin,h1); self.l1=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True)
        self.fc2=nn.Linear(h1,h2);  self.l2=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True)
        self.head=nn.Linear(h2,9)
    def forward(self,s):
        m1=self.l1.init_leaky(); m2=self.l2.init_leaky(); acc=0.
        for t in range(s.shape[1]):
            c1,m1=self.l1(self.fc1(s[:,t]),m1)
            c2,m2=self.l2(self.fc2(c1),m2); acc=acc+c2
        return self.head(acc/s.shape[1])
def run(fold,epochs=25,bs=32,lr=2e-3):
    te=np.isin(SUBJ,FOLD[fold]); tr=~te
    xt=torch.tensor(Xt[tr]); et=torch.tensor(ERR[tr])
    xv=torch.tensor(Xt[te]); ev=ERR[te]
    net=Joint(enc.dim(C))
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=(9e9,None)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(xt))
        for i in range(0,len(perm),bs):
            idx=perm[i:i+bs]
            sc=net(enc(xt[idx]))
            loss=nn.functional.mse_loss(sc,torch.log1p(et[idx])) \
                 +0.5*nn.functional.cross_entropy(-sc,et[idx].argmin(1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
        sch.step(); net.eval(); outs=[]
        with torch.no_grad():
            for i in range(0,len(xv),128): outs.append(net(enc(xv[i:i+128])))
        sc=torch.cat(outs).numpy(); pk=ev[np.arange(len(ev)),sc.argmin(1)]
        if pk.mean()<best[0]: best=(float(pk.mean()),pk.copy())
    return best,sum(p.numel() for p in net.parameters())
pk=[]; t0=time.time()
for f in range(4):
    b,pn=run(f); pk.append(b[1]); print('fold%d MAE %.3f'%(f,b[0]),flush=True)
pk=np.concatenate(pk)
print('\n=== 공동 채점(파형만, 특징 없음) === MAE %.2f  1BPM %.0f%%  파라미터 %d  %.0f초'%(
    pk.mean(),100*(pk<=1).mean(),pn,time.time()-t0))
json.dump(dict(mae=float(pk.mean()),p1=float((pk<=1).mean()),params=pn),
          open('joint.json','w'),ensure_ascii=False)
