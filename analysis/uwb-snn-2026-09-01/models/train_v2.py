# -*- coding: utf-8 -*-
"""제대로 된 분할 — 피험자를 학습/검증/평가 3분할.
   에폭은 검증셋으로만 고르고(early stopping), 평가셋은 마지막에 한 번만 본다.

근거:
- 3분할: 이전 코드는 평가셋에서 최고 에폭을 골라 결과가 부풀려졌음. 표준 절차로 교정.
- MAXEP=60 / PATIENCE=10: 상한을 넉넉히 두고 검증셋이 10에폭 동안 나아지지 않으면 중단.
  에폭 수를 사람이 정하지 않고 데이터가 정하게 함.
- 검증셋 = 다음 폴드: 폴드를 하나 더 쪼개지 않고 회전시켜, 피험자 수가 적은 상황에서
  학습 데이터를 최대한 남김.
"""
import numpy as np, torch, torch.nn as nn, json, time
import snntorch as snn
from snntorch import surrogate
from enc2 import ENC
torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(8)
MAXEP=60; PATIENCE=10; BS=32; LR=2e-3; H1=96; H2=48; FOLDS=4; BETA=0.9

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
    return X,((AUX-mu)/sd).astype(np.float32),EST,Y,ERR,D['subj']

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
    def forward(s,x,a): return s.head(torch.cat([s.body(x.squeeze(-1)),a],1)).squeeze(1),0.0

def evaluate(net,enc,x,a,err,C,T,A):
    net.eval(); outs=[]; rs=[]
    with torch.no_grad():
        for i in range(0,len(x),64):
            b=len(x[i:i+64]); xb=x[i:i+64].reshape(b*C,T,1)
            xb=enc(xb) if enc else xb
            sc,rt=net(xb,a[i:i+64].reshape(b*C,A)); outs.append(sc.reshape(b,C)); rs.append(rt)
    sc=torch.cat(outs).numpy()
    return err[np.arange(len(err)),sc.argmin(1)], float(np.mean(rs))

def run(mode,kind,encname):
    X,AUX,EST,Y,ERR,SUBJ=load(mode)
    N,C,T=X.shape; A=AUX.shape[2]
    SUB=sorted(set(SUBJ.tolist())); FD=[SUB[i::FOLDS] for i in range(FOLDS)]
    enc=[e for e in ENC if e.name==encname][0] if kind=='SNN' else None
    pick=np.zeros(N); spikes=[]; eps=[]
    for f in range(FOLDS):
        te_s=FD[f]; va_s=FD[(f+1)%FOLDS]
        te=np.isin(SUBJ,te_s); va=np.isin(SUBJ,va_s); tr=~(te|va)
        xt=torch.tensor(X[tr]); at=torch.tensor(AUX[tr]); et=torch.tensor(ERR[tr])
        xv=torch.tensor(X[va]); av=torch.tensor(AUX[va]); ev=ERR[va]
        xs=torch.tensor(X[te]); as_=torch.tensor(AUX[te]); es=ERR[te]
        net=SNNNet(enc.dim(1),A) if kind=='SNN' else ANNNet(T,A)
        opt=torch.optim.Adam(net.parameters(),lr=LR,weight_decay=1e-4)
        best_va=9e9; best_state={k:v.clone() for k,v in net.state_dict().items()}; best_ep=0; bad=0
        for ep in range(MAXEP):
            net.train(); perm=torch.randperm(len(xt))
            for i in range(0,len(perm),BS):
                idx=perm[i:i+BS]; b=len(idx)
                xb=xt[idx].reshape(b*C,T,1); xb=enc(xb) if enc else xb
                sc,_=net(xb,at[idx].reshape(b*C,A)); sc=sc.reshape(b,C)
                loss=nn.functional.mse_loss(sc,torch.log1p(et[idx])) \
                     +0.5*nn.functional.cross_entropy(-sc,et[idx].argmin(1))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),5.); opt.step()
            pv,_=evaluate(net,enc,xv,av,ev,C,T,A)
            if pv.mean()<best_va-1e-4:
                best_va=float(pv.mean()); best_ep=ep+1; bad=0
                best_state={k:v.clone() for k,v in net.state_dict().items()}
            else:
                bad+=1
                if bad>=PATIENCE: break
        net.load_state_dict(best_state)
        ps,rt=evaluate(net,enc,xs,as_,es,C,T,A)
        pick[te]=ps; spikes.append(rt); eps.append(best_ep)
    base=np.abs(np.median(EST[:,[0,3,6]],1)-Y)
    return dict(mode=mode,kind=kind,enc=encname,mae=float(pick.mean()),
                p1=float((pick<=1).mean()),p25=float((pick<=2.5).mean()),
                spike=float(np.mean(spikes)),params=sum(p.numel() for p in net.parameters()),
                stop_epochs=eps,base_mae=float(base.mean()),base_p1=float((base<=1).mean()),
                n=int(N),subj=len(SUB))

if __name__=='__main__':
    res=[]
    for mode in ['marker','landmark']:
        for kind,encname in [('ANN','-')]+[('SNN',e) for e in ['rate','sf','pop','delta','direct']]:
            t0=time.time(); r=run(mode,kind,encname); r['secs']=round(time.time()-t0)
            res.append(r)
            print('%-9s %-4s %-7s MAE %.2f 1BPM %.0f%% 정지에폭%s (%ds)'%(
                mode,kind,encname,r['mae'],100*r['p1'],r['stop_epochs'],r['secs']),flush=True)
            json.dump(res,open('train_v2.json','w'),ensure_ascii=False)
    print('\n완료')
