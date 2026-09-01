# -*- coding: utf-8 -*-
"""학습 가능성 점검 — 후보별 특징으로 '어느 후보가 맞는지' 예측되는가
   SNN 이전에, 신호 자체에 선택 단서가 있는지 확인하는 통과 조건."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
D=np.load('ds_rate.npz',allow_pickle=True)
F=D['feat']; EST=D['est']; Y=D['y']; ERR=D['err']; SUBJ=D['subj']
N,C,_=F.shape
# 후보 간 관계 특징 추가 (라벨 사용 안 함)
med=np.median(EST,axis=1,keepdims=True)
dev=np.abs(EST-med)
cons=np.stack([[np.sum(np.abs(EST[i]-EST[i,c])<=1.0)-1 for c in range(C)] for i in range(N)]).astype(np.float32)
iqr=(np.percentile(EST,75,axis=1)-np.percentile(EST,25,axis=1))[:,None]*np.ones((1,C))
XF=np.concatenate([F,EST[...,None],dev[...,None],cons[...,None],iqr[...,None]],axis=2)
NAMES=list(D['featname'])+['추정값','중앙값과의차','합의수','후보분산']
SUB=sorted(set(SUBJ.tolist())); FOLD=[SUB[i::4] for i in range(4)]
picked=np.zeros(N); base_med3=np.abs(np.median(EST[:,[0,3,6]],axis=1)-Y)
base_med9=np.abs(np.median(EST,axis=1)-Y)
imp=np.zeros(XF.shape[2])
for f in range(4):
    te=np.isin(SUBJ,FOLD[f]); tr=~te
    xt=XF[tr].reshape(-1,XF.shape[2]); yt=np.log1p(ERR[tr].reshape(-1))
    m=HistGradientBoostingRegressor(max_depth=4,max_iter=250,learning_rate=.07,
                                    l2_regularization=1.0,random_state=0).fit(xt,yt)
    p=m.predict(XF[te].reshape(-1,XF.shape[2])).reshape(te.sum(),C)
    idx=np.argmin(p,axis=1)
    picked[te]=ERR[te][np.arange(te.sum()),idx]
print('창 %d개 · 피험자 %d명 · 후보 %d개'%(N,len(SUB),C))
def rep(t,e): print('%-24s MAE %5.2f  1BPM이내 %3.0f%%  2.5이내 %3.0f%%'%(
    t,e.mean(),100*np.mean(e<=1),100*np.mean(e<=2.5)))
rep('선형 3대 중앙값(기준선)',base_med3)
rep('9후보 중앙값',base_med9)
rep('학습 선택기',picked)
rep('9후보 최선(상한)',ERR.min(1))
import collections
print('\n구간별 (기준선 → 학습 선택기)')
for b in ['평소 호흡','느린 호흡','회복 호흡','운동 후 호흡']:
    m=D['blk']==b
    print('  %-12s %5.2f → %5.2f'%(b,base_med3[m].mean(),picked[m].mean()))
