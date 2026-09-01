# -*- coding: utf-8 -*-
"""선배 요청: 평소 호흡 구간의 BIOPAC 호흡률이 비정상인 피험자 골라내기"""
import numpy as np, json, os
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import uniform_filter1d
W='_work'
AN=json.load(open('anchors.json'))
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x)
def apnea_c(env,fsb,lo,hi):
    k=int(22*fsb); ma=uniform_filter1d(env,k)
    i0=max(int(lo*fsb),k//2); i1=min(int(hi*fsb),len(ma)-k//2)
    if i1-i0<k: return None
    i=int(np.argmin(ma[i0:i1]))+i0
    return i/fsb, float(ma[i]/(np.median(ma)+1e-12))
out=[]
for s,v in sorted(AN.items()):
    p=os.path.join(W,s+'_mot.npz')
    if not os.path.exists(p): continue
    z=np.load(p); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    x=bp(rsp,0.08,0.8,fsb); env=uniform_filter1d(np.abs(x),int(4*fsb))
    r=apnea_c(env,fsb,max(v['ex_center']-260,20),v['ex_center']-40)
    if r is None or r[1]>0.5: continue
    ap=r[0]; i0=int((ap-140)*fsb); i1=int((ap-78)*fsb)
    if i0<0 or i1>len(x): continue
    seg=x[i0:i1]
    pk,_=find_peaks(seg,distance=int(1.2*fsb),prominence=np.std(seg)*0.5)
    rr=len(pk)/((i1-i0)/fsb/60.0)
    # 진폭 변동계수 — 벨트가 헐거우면 들쭉날쭉
    iv=np.diff(pk)/fsb
    cv=float(np.std(iv)/np.mean(iv)) if len(iv)>3 else np.nan
    out.append((s,rr,cv))
out.sort(key=lambda t:t[1])
print('%-10s %8s %8s  %s'%('피험자','호흡률','주기변동','판정'))
for s,rr,cv in out:
    flag=''
    if rr>=25: flag='◀ 25 초과 — 벨트/피험자 확인 요청'
    elif rr<10: flag='◀ 10 미만 — 확인 권장'
    if cv==cv and cv>0.45: flag=(flag+' / 주기 불규칙') if flag else '◀ 주기 불규칙'
    print('%-10s %8.1f %8.2f  %s'%(s,rr,cv,flag))
a=np.array([r for _,r,_ in out])
print('\nn=%d  중앙값 %.1f  범위 %.1f~%.1f'%(len(a),np.median(a),a.min(),a.max()))
print('선배 기준(16~22) 밖: %d명 / 25 초과: %d명'%(((a<16)|(a>22)).sum(),(a>25).sum()))
