import json,time,numpy as np
from train_B import run, FOLDS, SUBJ
from enc2 import ENC
res={}; t0=time.time()
for e in ENC:
    k='enc:'+e.name; res[k]=[]
    for f in range(3):
        r=run(e,f,readout='mem',epochs=20); r['enc']=e.name; r['readout']='mem'; res[k].append(r)
        print('%-12s fold%d best %.3f final %.3f spike %.3f'%(k,f,r['best'],r['hist'][-1]['val_acc'],r['hist'][-1]['spike']),flush=True)
    a=[x['hist'][-1]['val_acc'] for x in res[k]]
    print('>>> %s 평균 %.1f%% (±%.1f)\n'%(k,np.mean(a)*100,np.std(a)*100),flush=True)
    json.dump(res,open('results_B.json','w'),ensure_ascii=False)
best=max(ENC,key=lambda e: np.mean([x['hist'][-1]['val_acc'] for x in res['enc:'+e.name]]))
print('최고 인코딩:',best.name,flush=True)
for ro in ['count','first']:
    k='dec:'+ro; res[k]=[]
    for f in range(3):
        r=run(best,f,readout=ro,epochs=20); r['enc']=best.name; r['readout']=ro; res[k].append(r)
        print('%-12s fold%d best %.3f final %.3f'%(k,f,r['best'],r['hist'][-1]['val_acc']),flush=True)
    a=[x['hist'][-1]['val_acc'] for x in res[k]]
    print('>>> %s (%s) 평균 %.1f%% (±%.1f)\n'%(k,best.name,np.mean(a)*100,np.std(a)*100),flush=True)
    json.dump(res,open('results_B.json','w'),ensure_ascii=False)
json.dump(res,open('results_B.json','w'),ensure_ascii=False)
print('총 %.1f분'%((time.time()-t0)/60))
