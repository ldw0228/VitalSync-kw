# -*- coding: utf-8 -*-
import json, time, numpy as np
from train_A import run, FOLDS, SUBJ
from enc2 import ENC
res={}; t00=time.time()
# 1단계: 인코딩 비교 (디코딩=막전위 누적 고정)
for e in ENC:
    key='enc:'+e.name; res[key]=[]
    for f in range(3):
        t0=time.time(); r=run(e,f,readout='mem',epochs=20)
        r['secs']=round(time.time()-t0,1); r['enc']=e.name; r['readout']='mem'
        res[key].append(r)
        print('%-12s fold%d  best %.3f  final %.3f  spike %.3f  %.0fs'%(
            key,f,r['best'],r['hist'][-1]['val_acc'],r['hist'][-1]['spike'],r['secs']),flush=True)
    a=[x['hist'][-1]['val_acc'] for x in res[key]]
    print('>>> %s  평균 %.1f%% (±%.1f)\n'%(key,np.mean(a)*100,np.std(a)*100),flush=True)
    json.dump(res,open('results_A.json','w'),ensure_ascii=False)
# 2단계: 최고 인코딩으로 디코딩 비교
best=max(ENC,key=lambda e: np.mean([x['hist'][-1]['val_acc'] for x in res['enc:'+e.name]]))
print('최고 인코딩:',best.name,flush=True)
for ro in ['count','first']:
    key='dec:'+ro; res[key]=[]
    for f in range(3):
        t0=time.time(); r=run(best,f,readout=ro,epochs=20)
        r['secs']=round(time.time()-t0,1); r['enc']=best.name; r['readout']=ro
        res[key].append(r)
        print('%-12s fold%d  best %.3f  final %.3f  %.0fs'%(key,f,r['best'],r['hist'][-1]['val_acc'],r['secs']),flush=True)
    a=[x['hist'][-1]['val_acc'] for x in res[key]]
    print('>>> %s (%s)  평균 %.1f%% (±%.1f)\n'%(key,best.name,np.mean(a)*100,np.std(a)*100),flush=True)
    json.dump(res,open('results_A.json','w'),ensure_ascii=False)
json.dump(res,open('results_A.json','w'),ensure_ascii=False)
print('총 %.1f분'%((time.time()-t00)/60))
