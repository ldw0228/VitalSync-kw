# -*- coding: utf-8 -*-
"""실험 1·2 한정 동기화 판정 — 레이더 3대의 합의로 등급을 매김"""
import numpy as np, json, os, glob
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
W='_work'
LM=json.load(open('landmarks.json')); FSR=10.0
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def dip(e,fs,lo,hi,win=20.0):
    k=int(win*fs); ma=uniform_filter1d(e,k)
    i0=max(int(lo*fs),k//2); i1=min(int(hi*fs),len(ma)-k//2)
    if i1-i0<k: return None
    i=int(np.argmin(ma[i0:i1]))+i0
    return i/fs, float(ma[i]/(np.median(ma)+1e-12))
OUT=[]
for p in sorted(glob.glob(os.path.join(W,'*_iq.npz'))):
    s=os.path.basename(p)[:-7]; pm=os.path.join(W,s+'_mot.npz')
    if s not in LM or not os.path.exists(pm):
        OUT.append(dict(s=s,grade='실패',n=0,note='실험1 회전 미검출')); continue
    end1=LM[s]['blocks'][2][1]; lo,hi=end1+100,end1+260
    z=np.load(p); A=z['re'].astype(np.float32)+1j*z['im'].astype(np.float32)
    zm=np.load(pm); rsp=zm['rsp'].astype(float); fsb=float(zm['fsb'])
    eb=uniform_filter1d(np.abs(bp(rsp,0.08,0.8,fsb)),int(4*fsb))
    B=dip(eb,fsb,lo-30,hi+30)
    if B is None: OUT.append(dict(s=s,grade='실패',n=0,note='구간 부족')); continue
    apb,db=B
    per=[]
    j0,j1=int(lo*FSR),min(int(hi*FSR),A.shape[1])
    for r in range(A.shape[0]):
        seg=A[r,j0:j1]-A[r,j0:j1].mean(0,keepdims=True)   # 탐색창 안에서 가슴 bin 선택
        e=bp(np.abs(seg),0.1,0.6,FSR).std(0); kb=int(np.argmax(e))
        c=seg[:,max(kb-1,0):kb+2].mean(1)
        xy=np.stack([c.real,c.imag],1); xy-=xy.mean(0)
        _,_,vt=np.linalg.svd(xy,full_matrices=False)
        ev=uniform_filter1d(np.abs(bp(xy@vt[0],0.1,0.6,FSR)),int(4*FSR))
        d=dip(ev,FSR,0,(j1-j0)/FSR)
        if d is None: continue
        ap,dep=d; ap=lo+ap; off=apb-ap
        per.append(dict(r=r+1,ap=ap,dep=dep,off=off,ok=(dep<0.5 and -20<=off<=15)))
    good=[x for x in per if x['ok']]
    n=len(good)
    spread=(max(x['ap'] for x in good)-min(x['ap'] for x in good)) if n>=2 else 0.0
    if n>=2 and spread<=10: grade='확실'
    elif n>=2: grade='주의'; 
    elif n==1: grade='주의'
    else: grade='실패'
    OUT.append(dict(s=s,grade=grade,n=n,spread=round(spread,1),
                    ap=round(float(np.median([x['ap'] for x in good])),1) if n else None,
                    off=round(float(np.median([x['off'] for x in good])),1) if n else None,
                    dep=round(float(np.median([x['dep'] for x in good])),2) if n else None,
                    bio_dep=round(db,2),
                    note='' if n else '무호흡 함몰 미검출',
                    per=per))
print('%-10s %-5s %4s %7s %8s %7s  %s'%('피험자','판정','합의','무호흡','offset','함몰','비고'))
for o in OUT:
    print('%-10s %-5s %4s %7s %8s %7s  %s'%(o['s'],o['grade'],'%d/3'%o['n'],
        ('%.1f'%o['ap']) if o.get('ap') else '-',
        ('%+.1f'%o['off']) if o.get('off') is not None else '-',
        ('%.2f'%o['dep']) if o.get('dep') is not None else '-', o.get('note','')))
from collections import Counter
c=Counter(o['grade'] for o in OUT)
print('\n확실 %d명 · 주의 %d명 · 실패 %d명  (총 %d)'%(c['확실'],c['주의'],c['실패'],len(OUT)))
offs=[o['off'] for o in OUT if o.get('off') is not None]
print('offset 범위 %.1f ~ %.1f초 (중앙값 %.1f)'%(min(offs),max(offs),np.median(offs)))
json.dump(OUT,open('sync12c.json','w'),ensure_ascii=False,indent=1)
