# -*- coding: utf-8 -*-
"""두 자르기 방식 × (ANN, SNN×인코딩 5종) 전체 비교
   과제: 9개 후보 중 어느 추정을 믿을지 고르기.  평가: 고른 후보의 호흡수 오차"""
import numpy as np, torch, torch.nn as nn, json, time, sys, itertools
import snntorch as snn
from snntorch import surrogate
from enc2 import ENC
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
EPOCHS=15; BS=32; LR=2e-3; H1=96; H2=48; FOLDS=4; BETA=0.9
def load(mode):
    D=np.load('ds_%s.npz'%mode,allow_pickle=True)
    X=D['X'].astype(np.float32); FE=D['feat'].astype(np.float32)
    EST=D['est'].astype(np.float32); Y=D['y'].astype(np.float32); ERR=D['err'].astype(np.float32)
    N,C,T=X.shape; DS=3
    X=X[:,:,:(T//DS)*DS].reshape(N,C,-1,DS).mean(-1)
    med=np.median(EST,1,keepdims=True); dev=np.abs(EST-med)
    cons=np.stack([[np.sum(np.abs(EST[i]-EST[i,c])<=1.)-1 for c in range(C)] for i in range(N)]).astype(np.float32)
    AUX=np.concatenate([FE,EST[...,None],dev[...,None],cons[...,None]],2)
    mu=AUX.reshape(-1,AUX.shape[2]).mean(0); sd=AUX.reshape(-1,AUX.shape[2]).std(0)+1e-6
    return X,((AUX-mu)/sd).astype(np.float32),EST,Y,ERR,D['subj'],D['blk']
class SNNNet(nn.Module):
    def __init__(s,fin,aux):
        super().__init__(); sg=surrogate.atan()
        s.fc1=nn.Linear(fin,H1); s.l1=snn.Leaky(beta=BETA,spike_grad=sg,learn_beta=True)
        s.fc2=nn.Linear(H1,H2);  s.l2=snn.Leaky(beta=BETA,spike_grad=sg,learn_beta=True)
        s.head=nn.Sequential(nn.Linear(H2+aux,48),nn.ReLU(),nn.Linear(48,1))
    def forward(s,x,a):
        m1=s.l1.init_leaky(); m2=s.l2.init_leaky(); acc=0.; spk=0.
        for t in range(x.shape[1]):
            c1,m1=s.l1(s.fc1(x[:,t]),m1); c2,m2=s.l2(s.fc2(c1),m2)
            acc=acc+c2; spk=spk+c1.mean()+c2.mean()
        return s.head(torch.cat([acc/x.shape[1],a],1)).squeeze(1), float(spk.detach())/(2*x.shape[1])
class ANNNet(nn.Module):
    def __init__(s,T,aux):
        super().__init__()
        s.body=nn.Sequential(nn.Linear(T,H1),nn.ReLU(),nn.Linear(H1,H2),nn.ReLU())
        s.head=nn.Sequential(nn.Linear(H2+aux,48),nn.ReLU(),nn.Linear(48,1))
    def forward(s,x,a):
        return s.head(torch.cat([s.body(x.squeeze(-1)),a],1)).squeeze(1), 0.0
def run(mode,kind,encname):
    X,AUX,EST,Y,ERR,SUBJ,BLK=load(mode)
    N,C,T=X.shape; A=AUX.shape[2]
    SUB=sorted(set(SUBJ.tolist())); FD=[SUB[i::FOLDS] for i in range(FOLDS)]
    enc=[e for e in ENC if e.name==encname][0] if kind=='SNN' else None
    pick=np.zeros(N); spikes=[]
    for f in range(FOLDS):
        te=np.isin(SUBJ,FD[f]); tr=~te
        xt=torch.tensor(X[tr]); at=torch.tensor(AUX[tr]); et=torch.tensor(ERR[tr])
        xv=torch.tensor(X[te]); av=torch.tensor(AUX[te]); ev=ERR[te]
        net=SNNNet(enc.dim(1),A) if kind=='SNN' else ANNNet(T,A)
        opt=torch.optim.Adam(net.parameters(),lr=LR,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
        best=(9e9,None,0.)
        for ep in range(EPOCHS):
            net.train(); perm=torch.randperm(len(xt))
            for i in range(0,len(perm),BS):
                idx=perm[i:i+BS]; b=len(idx)
                xb=xt[idx].reshape(b*C,T,1); xb=enc(xb) if enc else xb
                sc,_=net(xb,at[idx].reshape(b*C,A)); sc=sc.reshape(b,C)
                loss=nn.functional.mse_loss(sc,torch.log1p(et[idx])) \
                     +0.5*nn.functional.cross_entropy(-sc,et[idx].argmin(1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
            sch.step(); net.eval(); outs=[]; rs=[]
            with torch.no_grad():
                for i in range(0,len(xv),64):
                    b=len(xv[i:i+64]); xb=xv[i:i+64].reshape(b*C,T,1)
                    xb=enc(xb) if enc else xb
                    sc,rt=net(xb,av[i:i+64].reshape(b*C,A)); outs.append(sc.reshape(b,C)); rs.append(rt)
            sc=torch.cat(outs).numpy(); pk=ev[np.arange(len(ev)),sc.argmin(1)]
            if pk.mean()<best[0]: best=(float(pk.mean()),pk.copy(),float(np.mean(rs)))
        pick[te]=best[1]; spikes.append(best[2])
    base=np.abs(np.median(EST[:,[0,3,6]],1)-Y)
    return dict(mode=mode,kind=kind,enc=encname,mae=float(pick.mean()),
                p1=float((pick<=1).mean()),p25=float((pick<=2.5).mean()),
                spike=float(np.mean(spikes)),params=sum(p.numel() for p in net.parameters()),
                base_mae=float(base.mean()),base_p1=float((base<=1).mean()),n=int(N),
                subj=len(SUB))
res=[]
for mode in ['marker','landmark']:
    for kind,encname in [('ANN','-')]+[('SNN',e) for e in ['rate','sf','pop','delta','direct']]:
        t0=time.time(); r=run(mode,kind,encname); r['secs']=round(time.time()-t0)
        res.append(r); print('%-9s %-4s %-7s MAE %.2f 1BPM %.0f%% (%ds)'%(
            mode,kind,encname,r['mae'],100*r['p1'],r['secs']),flush=True)
        json.dump(res,open('train_all.json','w'),ensure_ascii=False)
print('\n완료')
