# -*- coding: utf-8 -*-
"""마커로 자른 것 vs 지형지물로 자른 것 — 같은 고전 추정기로 나란히 비교"""
import numpy as np, json, os, csv
from scipy.signal import butter, filtfilt, welch
W='_work'
LM=json.load(open('landmarks.json')); SY={o['s']:o for o in json.load(open('sync12_final.json'))}
FSR=10.0; WIN=30.0; HOP=5.0
BLK=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('회복 호흡',16,78),('운동 후 호흡',172,234)]
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def rate(x,fs):
    f,P=welch(x,fs=fs,nperseg=min(len(x),int(WIN*fs)),nfft=1<<14)
    m=(f>0.06)&(f<0.80); f,P=f[m],P[m]; i=int(np.argmax(P))
    if 0<i<len(P)-1:
        a,b,c=np.log(P[i-1]+1e-30),np.log(P[i]+1e-30),np.log(P[i+1]+1e-30)
        d=np.clip(0.5*(a-c)/(a-2*b+c+1e-30),-1,1); return float((f[i]+d*(f[1]-f[0]))*60)
    return float(f[i]*60)
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
def eval_cut(A,xb,fsb,apr,apb):
    """무호흡 중심(레이더 apr / BIOPAC apb) 기준으로 잘라 오차 집계"""
    es=[]
    for nm,a0,a1 in BLK:
        t=a0
        while t+WIN<=a1:
            j0,j1=int((apr+t)*FSR),int((apr+t+WIN)*FSR)
            i0,i1=int((apb+t)*fsb),int((apb+t+WIN)*fsb)
            if j0>=0 and j1<=A.shape[1] and i0>=0 and i1<=len(xb):
                y=rate(xb[i0:i1],fsb); rr=[]
                for r in range(3):
                    s=A[r,j0:j1]-A[r,j0:j1].mean(0,keepdims=True)
                    e=bp(np.abs(s),0.1,0.6,FSR).std(0); k=int(np.argmax(e))
                    c=s[:,max(k-1,0):k+2].mean(1)
                    xy=np.stack([c.real,c.imag],1); xy-=xy.mean(0)
                    _,_,vt=np.linalg.svd(xy,full_matrices=False)
                    rr.append(rate(bp(xy@vt[0],0.06,0.80,FSR),FSR))
                es.append(abs(float(np.median(rr))-y))
            t+=HOP
    return es
rows=[]; EA=[]; EB=[]
for s,o in sorted(SY.items()):
    if o['grade']=='실패' or s not in LM: continue
    pm=os.path.join(W,s+'_mot.npz'); pi=os.path.join(W,s+'_iq.npz')
    if not(os.path.exists(pm) and os.path.exists(pi)): continue
    z=np.load(pm); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    xb=bp(rsp,0.06,0.80,fsb)
    zi=np.load(pi); A=zi['re'].astype(np.float32)+1j*zi['im'].astype(np.float32)
    mk=markers(rsp,fsb)
    # B: 지형지물 (현재)
    eb=eval_cut(A,xb,fsb,o['ap'],o['ap']+o['off'])
    # A: 마커 — 3번째 마커를 실험2 시작으로, 대본상 무호흡 중심 = +140초
    ea=[]
    if len(mk)>=3:
        apb_m=mk[2]+140.0
        apr_m=apb_m-o['off']          # 시간축 변환은 동일 offset 사용 (경계 출처만 비교)
        ea=eval_cut(A,xb,fsb,apr_m,apb_m)
    rows.append((s,len(mk),
                 float(np.mean(ea)) if ea else np.nan, len(ea),
                 float(np.mean(eb)) if eb else np.nan, len(eb),
                 (mk[2] if len(mk)>=3 else np.nan),
                 o['ap']+o['off']-140.0))
    EA+=ea; EB+=eb
print('%-10s %5s %10s %10s %12s %12s'%('피험자','마커','마커기준','지형지물','마커#3(초)','우리 실험2시작'))
for s,n,ma,na,mb,nb,m3,ours in rows:
    print('%-10s %5d %10s %10.2f %12s %12.1f'%(s,n,('%.2f'%ma) if ma==ma else '-',mb,
          ('%.1f'%m3) if m3==m3 else '-',ours))
EA=np.array(EA); EB=np.array(EB)
print('\n전체  마커기준  MAE %.2f  1BPM %.0f%%  (창 %d)'%(EA.mean(),100*np.mean(EA<=1),len(EA)))
print('전체  지형지물  MAE %.2f  1BPM %.0f%%  (창 %d)'%(EB.mean(),100*np.mean(EB<=1),len(EB)))
json.dump([list(r) for r in rows],open('ab_marker.json','w'),ensure_ascii=False)
