# -*- coding: utf-8 -*-
"""지형지물로 찾은 실험 1·2 위치가 BIOPAC 마커와 맞는지 검증"""
import numpy as np, json, os
W='_work'
LM=json.load(open('landmarks.json'))
SY={o['s']:o for o in json.load(open('sync12_final.json'))}
def markers(rsp,fsb,thr=8.5,merge=4.0):
    ab=rsp>thr; d=np.diff(np.r_[0,ab.astype(int),0])
    on=np.where(d==1)[0]; off=np.where(d==-1)[0]-1
    mk=[]
    for a,b in zip(on,off):
        seg=rsp[a:b+1]; mk.append((a+int(np.argmax(seg)))/fsb)
    if not mk: return []
    mk=sorted(mk); out=[]; grp=[mk[0]]
    for x in mk[1:]:
        if x-grp[-1]<=merge: grp.append(x)
        else: out.append(float(np.mean(grp))); grp=[x]
    out.append(float(np.mean(grp))); return out
rows=[]
for s,o in sorted(SY.items()):
    if o['grade']=='실패' or s not in LM: continue
    p=os.path.join(W,s+'_mot.npz')
    if not os.path.exists(p): continue
    z=np.load(p); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    mk=markers(rsp,fsb)
    if len(mk)<4: rows.append((s,len(mk),None,None,None)); continue
    off=o['off']                              # bio = radar + off
    bl=LM[s]['blocks']
    s1=bl[0][0]+off; e1=bl[2][1]+off          # 실험1 시작·종료 (BIOPAC 시간)
    ap=o['ap']+off                            # 무호흡 중심 (BIOPAC 시간)
    near=lambda t: min(mk,key=lambda m:abs(m-t))
    rows.append((s,len(mk),near(s1)-s1,near(e1)-e1,ap))
print('%-10s %5s %12s %12s'%('피험자','마커수','실험1 시작','실험1 종료'))
d1=[];d2=[]
for s,n,a,b,ap in rows:
    if a is None: print('%-10s %5d %12s %12s'%(s,n,'-','-')); continue
    d1.append(a); d2.append(b)
    print('%-10s %5d %+11.1f초 %+11.1f초'%(s,n,a,b))
d1=np.array(d1); d2=np.array(d2)
print('\n지형지물 위치와 가장 가까운 마커의 차이')
print('  실험1 시작: 중앙값 %+.1f초, |차이| 5초 이내 %d/%d명'%(np.median(d1),np.sum(np.abs(d1)<5),len(d1)))
print('  실험1 종료: 중앙값 %+.1f초, |차이| 5초 이내 %d/%d명'%(np.median(d2),np.sum(np.abs(d2)<5),len(d2)))
