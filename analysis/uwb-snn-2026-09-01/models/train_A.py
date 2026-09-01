# -*- coding: utf-8 -*-
import numpy as np, torch, torch.nn as nn, time, json
import snntorch as snn
from snntorch import surrogate
from enc2 import ENC
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)

D=np.load('dataset_A.npz',allow_pickle=True)
X=D['X'].astype(np.float32); y=D['y']; g=D['g']; rad=D['rad']
N,T,F=X.shape; NC=3
SUBJ=sorted(set(g.tolist()))
FOLDS=[SUBJ[i::3] for i in range(3)]      # 피험자 3-fold

class SNNNet(nn.Module):
    """readout: 'mem'(막전위 누적) | 'count'(발화수) | 'first'(최초발화) | 'popvec'(모집단 벡터)"""
    def __init__(self, fin, nc=NC, h1=128, h2=64, beta=0.9, readout='mem', npop=9):
        super().__init__(); sg=surrogate.atan(); self.readout=readout; self.npop=npop
        nout = npop if readout=='popvec' else nc
        self.fc1=nn.Linear(fin,h1); self.lif1=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True)
        self.fc2=nn.Linear(h1,h2);  self.lif2=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True)
        self.fc3=nn.Linear(h2,nout)
        rm='none' if readout=='mem' else 'subtract'
        self.lif3=snn.Leaky(beta=beta,spike_grad=sg,learn_beta=True,reset_mechanism=rm)
    def forward(self,s):
        m1=self.lif1.init_leaky(); m2=self.lif2.init_leaky(); m3=self.lif3.init_leaky()
        acc=0.; cnt=0.; firstw=0.; wsum=0.; spk=0.
        Tt=s.shape[1]
        for t in range(Tt):
            c1,m1=self.lif1(self.fc1(s[:,t]),m1)
            c2,m2=self.lif2(self.fc2(c1),m2)
            o3,m3=self.lif3(self.fc3(c2),m3)
            acc=acc+m3; cnt=cnt+o3
            w=(1.0-t/Tt)                      # 이른 발화에 큰 가중치 -> 최초발화 근사(미분가능)
            firstw=firstw+o3*w; wsum+=w
            spk=spk+c1.mean()+c2.mean()
        rate=float(spk.detach())/(2*Tt)
        if self.readout=='mem':   out=acc/Tt
        elif self.readout=='count': out=cnt/Tt*10.
        elif self.readout=='first': out=firstw/wsum*10.
        else:                      out=cnt/Tt*10.      # popvec: 로짓 대신 모집단 활동
        return out, rate

def run(enc, fold, readout='mem', epochs=22, bs=128, lr=2e-3, log=None):
    te=np.isin(g,FOLDS[fold]); tr=~te
    Xtr=torch.tensor(X[tr]); ytr=torch.tensor(y[tr]); Xte=torch.tensor(X[te]); yte=torch.tensor(y[te])
    net=SNNNet(enc.dim(F),readout=readout)
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    lossf=nn.CrossEntropyLoss(); hist=[]; best=(0.,None,None)
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(Xtr)); tc=0; tl=0.
        for i in range(0,len(perm),bs):
            idx=perm[i:i+bs]; xb=enc(Xtr[idx]); yb=ytr[idx]
            out,_=net(xb); loss=lossf(out,yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
            tl+=loss.item()*len(idx); tc+=(out.argmax(1)==yb).sum().item()
        sch.step()
        net.eval(); vc=0; preds=[]; rates=[]
        with torch.no_grad():
            for i in range(0,len(Xte),256):
                xb=enc(Xte[i:i+256]); out,rt=net(xb)
                p=out.argmax(1); preds.append(p); vc+=(p==yte[i:i+256]).sum().item(); rates.append(rt)
        preds=torch.cat(preds); va=vc/len(Xte)
        hist.append(dict(epoch=ep+1,train_acc=tc/len(Xtr),train_loss=tl/len(Xtr),
                         val_acc=va,spike=float(np.mean(rates))))
        if va>best[0]: best=(va,preds.numpy().copy(),yte.numpy().copy())
        if log: log('    ep%02d tr %.3f va %.3f'%(ep+1,tc/len(Xtr),va))
    return dict(fold=fold,test=FOLDS[fold],hist=hist,best=best[0],
                pred=best[1].tolist(),true=best[2].tolist(),
                params=sum(p.numel() for p in net.parameters()))
