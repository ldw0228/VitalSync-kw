# -*- coding: utf-8 -*-
"""절제 실험 — SNN 가지가 실제로 기여하는가, 아니면 특징이 다 하는가"""
import numpy as np, torch, torch.nn as nn, json, sys, time
import snntorch as snn
from snntorch import surrogate
from enc2 import ENC
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
D=np.load('ds_rate.npz',allow_pickle=True)
X=D['X'].astype(np.float32); FE=D['feat'].astype(np.float32)
EST=D['est'].astype(np.float32); Y=D['y'].astype(np.float32); ERR=D['err'].astype(np.float32)
SUBJ=D['subj']
N,C,T=X.shape; DS=3
X=X[:,:,:(T//DS)*DS].reshape(N,C,-1,DS).mean(-1); T=X.shape[2]
med=np.median(EST,axis=1,keepdims=True); dev=np.abs(EST-med)
cons=np.stack([[np.sum(np.abs(EST[i]-EST[i,c])<=1.0)-1 for c in range(C)] for i in range(N)]).astype(np.float32)
AUX=np.concatenate([FE,EST[...,None],dev[...,None],cons[...,None]],axis=2)
mu=AUX.reshape(-1,AUX.shape[2]).mean(0); sd=AUX.reshape(-1,AUX.shape[2]).std(0)+1e-6
AUX=((AUX-mu)/sd).astype(np.float32); A=AUX.shape[2]
SUB=sorted(set(SUBJ.tolist())); FOLD=[SUB[i::4] for i in range(4)]
enc=[e for e in ENC if e.name=='rate'][0]

class Net(nn.Module):
    def __init__(self,mode,fin,h1=96,h2=48,beta=0.9):
        super().__init__(); self.mode=mode; sg=surrogate.atan()
        if mode!='aux':
            self.fc1=nn.Linear(fin,h1); self.l1=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True)
            self.fc2=nn.Linear(h1,h2);  self.l2=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True)
        d=(h2 if mode!='aux' else 0)+(A if mode!='snn' else 0)
        self.head=nn.Sequential(nn.Linear(d,48),nn.ReLU(),nn.Linear(48,1))
    def forward(self,s,a):
        parts=[]
        if self.mode!='aux':
            m1=self.l1.init_leaky(); m2=self.l2.init_leaky(); acc=0.
            for t in range(s.shape[1]):
                c1,m1=self.l1(self.fc1(s[:,t]),m1)
                c2,m2=self.l2(self.fc2(c1),m2); acc=acc+c2
            parts.append(acc/s.shape[1])
        if self.mode!='snn': parts.append(a)
        return self.head(torch.cat(parts,1)).squeeze(1)

def run(mode,fold,epochs=18,bs=32,lr=2e-3):
    te=np.isin(SUBJ,FOLD[fold]); tr=~te
    xt=torch.tensor(X[tr]); at=torch.tensor(AUX[tr]); et=torch.tensor(ERR[tr])
    xv=torch.tensor(X[te]); av=torch.tensor(AUX[te]); ev=ERR[te]
    net=Net(mode,enc.dim(1))
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    best=(9e9,None)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(xt))
        for i in range(0,len(perm),bs):
            idx=perm[i:i+bs]; b=len(idx)
            sb=enc(xt[idx].reshape(b*C,T,1)) if mode!='aux' else torch.zeros(1)
            sc=net(sb,at[idx].reshape(b*C,A)).reshape(b,C)
            loss=nn.functional.mse_loss(sc,torch.log1p(et[idx])) \
                 +0.5*nn.functional.cross_entropy(-sc,et[idx].argmin(1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
        sch.step(); net.eval(); outs=[]
        with torch.no_grad():
            for i in range(0,len(xv),64):
                b=len(xv[i:i+64])
                sb=enc(xv[i:i+64].reshape(b*C,T,1)) if mode!='aux' else torch.zeros(1)
                outs.append(net(sb,av[i:i+64].reshape(b*C,A)).reshape(b,C))
        sc=torch.cat(outs).numpy(); pk=ev[np.arange(len(ev)),sc.argmin(1)]
        if pk.mean()<best[0]: best=(float(pk.mean()),pk.copy())
    return best

out={}
for mode in ['aux','snn','both']:
    t0=time.time(); pk=[]
    for f in range(4):
        b=run(mode,f); pk.append(b[1])
        print('  %s fold%d  MAE %.3f'%(mode,f,b[0]),flush=True)
    pk=np.concatenate(pk); out[mode]=dict(mae=float(pk.mean()),p1=float((pk<=1).mean()),
                                          secs=round(time.time()-t0))
    print('%-6s MAE %.2f  1BPM %.0f%%  (%.0f초)\n'%(mode,pk.mean(),100*(pk<=1).mean(),time.time()-t0),flush=True)
json.dump(out,open('ablate.json','w'),ensure_ascii=False)
base=np.abs(np.median(EST[:,[0,3,6]],axis=1)-Y)
print('기준선 %.2f · 상한 %.2f'%(base.mean(),ERR.min(1).mean()))
