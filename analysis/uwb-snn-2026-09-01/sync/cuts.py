# -*- coding: utf-8 -*-
"""피험자별로 실제 자른 구간 전체 — 레이더 시간 기준"""
import numpy as np, json, os, csv
W='_work'
LM=json.load(open('landmarks.json'))
SY={o['s']:o for o in json.load(open('sync12_final.json'))}
def markers(rsp,fsb,thr=8.5,merge=4.0):
    ab=rsp>thr; d=np.diff(np.r_[0,ab.astype(int),0])
    on=np.where(d==1)[0]; off=np.where(d==-1)[0]-1
    mk=[(a+int(np.argmax(rsp[a:b+1])))/fsb for a,b in zip(on,off)]
    if not mk: return []
    mk=sorted(mk); out=[]; grp=[mk[0]]
    for x in mk[1:]:
        if x-grp[-1]<=merge: grp.append(x)
        else: out.append(float(np.mean(grp))); grp=[x]
    out.append(float(np.mean(grp))); return out
B2=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('무호흡',-16,16),
    ('회복 호흡',16,78),('운동',94,156),('운동 후 호흡',172,234)]
rows=[]
for s,o in sorted(SY.items()):
    r={'피험자':s,'동기화 판정':o['grade']}
    if o['grade']=='실패' or s not in LM:
        r['비고']='실험1 회전 미검출'; rows.append(r); continue
    bl=LM[s]['blocks']; ap=o['ap']; off=o['off']
    p=os.path.join(W,s+'_mot.npz')
    mk=[]
    if os.path.exists(p):
        z=np.load(p); mk=markers(z['rsp'].astype(float),float(z['fsb']))
    r['마커 수']=len(mk)
    r['offset(BIOPAC-레이더,초)']=round(off,1)
    for i,(a,b) in enumerate(bl):
        r['실험1 블록%d 시작'%(i+1)]=round(a,1); r['실험1 블록%d 종료'%(i+1)]=round(b,1)
    r['무호흡 중심']=round(ap,1)
    for nm,a0,a1 in B2:
        r['실험2 %s 시작'%nm]=round(ap+a0,1); r['실험2 %s 종료'%nm]=round(ap+a1,1)
    if mk:
        near=lambda t: min(mk,key=lambda m:abs(m-t))
        r['마커 대조 실험1시작(초)']=round(near(bl[0][0]+off)-(bl[0][0]+off),1)
        r['마커 대조 실험1종료(초)']=round(near(bl[2][1]+off)-(bl[2][1]+off),1)
        r['마커 대조 무호흡(초)']=round(near(ap+off)-(ap+off),1)
        bad=abs(r['마커 대조 실험1시작(초)'])>30 or abs(r['마커 대조 무호흡(초)'])>30
        r['비고']='마커와 불일치 — 확인 필요' if bad else ''
    rows.append(r)
cols=['피험자','동기화 판정','마커 수','offset(BIOPAC-레이더,초)']
for i in (1,2,3): cols+=['실험1 블록%d 시작'%i,'실험1 블록%d 종료'%i]
cols+=['무호흡 중심']
for nm,_,_ in B2: cols+=['실험2 %s 시작'%nm,'실험2 %s 종료'%nm]
cols+=['마커 대조 실험1시작(초)','마커 대조 실험1종료(초)','마커 대조 무호흡(초)','비고']
with open('자른구간_전체.csv','w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader()
    for r in rows: w.writerow({c:r.get(c,'') for c in cols})
print('%-10s %-5s %5s %8s %10s %10s %10s %s'%('피험자','판정','마커','offset','실1시작','실1종료','무호흡중심','마커대조(시작/무호흡)'))
for r in rows:
    if r.get('동기화 판정')=='실패': print('%-10s %-5s  — 실험1 회전 미검출'%(r['피험자'],r['동기화 판정'])); continue
    print('%-10s %-5s %5s %8s %10s %10s %10s   %+.1f / %+.1f  %s'%(
        r['피험자'],r['동기화 판정'],r['마커 수'],r['offset(BIOPAC-레이더,초)'],
        r['실험1 블록1 시작'],r['실험1 블록3 종료'],r['무호흡 중심'],
        r.get('마커 대조 실험1시작(초)',0),r.get('마커 대조 무호흡(초)',0),r.get('비고','')))
