# -*- coding: utf-8 -*-
import numpy as np, torch, torch.nn as nn, json, time
from train_A import SNNNet, X, y, g, rad
from enc2 import Rate
F=X.shape[2]
def run_split(tr_mask, te_mask, enc, epochs=20, bs=128, lr=2e-3, readout='mem'):
    Xtr=torch.tensor(X[tr_mask]); ytr=torch.tensor(y[tr_mask])
    Xte=torch.tensor(X[te_mask]); yte=torch.tensor(y[te_mask])
    net=SNNNet(enc.dim(F),readout=readout)
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); lossf=nn.CrossEntropyLoss()
    best=0.
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(Xtr))
        for i in range(0,len(perm),bs):
            idx=perm[i:i+bs]; out,_=net(enc(Xtr[idx])); loss=lossf(out,ytr[idx])
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
        sch.step(); net.eval(); vc=0
        with torch.no_grad():
            for i in range(0,len(Xte),256):
                out,_=net(enc(Xte[i:i+256])); vc+=(out.argmax(1)==yte[i:i+256]).sum().item()
        best=max(best,vc/len(Xte))
    return best, vc/len(Xte)
if __name__=='__main__':
    enc=Rate(); res={}
    for nm,tr,te in [('r1 학습 → r3 검증 (등거리)', rad==1, rad==3),
                     ('r3 학습 → r1 검증 (등거리)', rad==3, rad==1),
                     ('r1+r3 학습 → r2 검증 (거리 다름)', (rad==1)|(rad==3), rad==2)]:
        t0=time.time(); b,f=run_split(tr,te,enc)
        res[nm]=dict(best=b,final=f,n_tr=int(tr.sum()),n_te=int(te.sum()))
        print('%-34s  최고 %.3f  최종 %.3f  (학습 %d / 검증 %d)  %.0fs'%(nm,b,f,tr.sum(),te.sum(),time.time()-t0),flush=True)
    json.dump(res,open('cross_radar.json','w'),ensure_ascii=False,indent=1)
