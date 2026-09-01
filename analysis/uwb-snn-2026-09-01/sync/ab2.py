# -*- coding: utf-8 -*-
"""마커 기반(HAI_동기화_종합.xlsx) vs 지형지물 기반 — 같은 조건으로 전체 비교"""
import numpy as np, json, os, csv
from scipy.signal import butter, filtfilt, welch
W='_work'
SY={o['s']:o for o in json.load(open('sync12_final.json'))}
XL=json.loads(open('xl_exp2.json').read())
FSR=10.0; WIN=30.0; HOP=5.0
BLK=[('평소 호흡',-140,-78),('느린 호흡',-78,-16),('회복 호흡',16,78),('운동 후 호흡',172,234)]
SCRIPT_APNEA=135.0          # 대본상 실험2 시작 -> 무호흡 중심까지
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def rate(x,fs):
    f,P=welch(x,fs=fs,nperseg=min(len(x),int(WIN*fs)),nfft=1<<14)
    m=(f>0.06)&(f<0.80); f,P=f[m],P[m]; i=int(np.argmax(P))
    if 0<i<len(P)-1:
        a,b,c=np.log(P[i-1]+1e-30),np.log(P[i]+1e-30),np.log(P[i+1]+1e-30)
        d=np.clip(0.5*(a-c)/(a-2*b+c+1e-30),-1,1); return float((f[i]+d*(f[1]-f[0]))*60)
    return float(f[i]*60)
def start(v):
    if not v or '?' in str(v): return None
    try: return float(str(v).split('~')[0])
    except Exception: return None
def run_cut(A,xb,fsb,apr,apb,tag,s,acc):
    for nm,a0,a1 in BLK:
        t=a0
        while t+WIN<=a1:
            j0,j1=int((apr+t)*FSR),int((apr+t+WIN)*FSR)
            i0,i1=int((apb+t)*fsb),int((apb+t+WIN)*fsb)
            if j0>=0 and j1<=A.shape[1] and i0>=0 and i1<=len(xb):
                y=rate(xb[i0:i1],fsb); rr=[]
                for r in range(3):
                    sg=A[r,j0:j1]-A[r,j0:j1].mean(0,keepdims=True)
                    e=bp(np.abs(sg),0.1,0.6,FSR).std(0); k=int(np.argmax(e))
                    c=sg[:,max(k-1,0):k+2].mean(1)
                    xy=np.stack([c.real,c.imag],1); xy-=xy.mean(0)
                    _,_,vt=np.linalg.svd(xy,full_matrices=False)
                    rr.append(rate(bp(xy@vt[0],0.06,0.80,FSR),FSR))
                acc.append(dict(s=s,blk=nm,t=t,tag=tag,y=y,est=rr,
                                err=abs(float(np.median(rr))-y)))
            t+=HOP
M=[];L=[];rows=[]
for s,o in sorted(SY.items()):
    num=int(s[1:3])
    pm=os.path.join(W,s+'_mot.npz'); pi=os.path.join(W,s+'_iq.npz')
    if not(os.path.exists(pm) and os.path.exists(pi)): continue
    z=np.load(pm); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    xb=bp(rsp,0.06,0.80,fsb)
    zi=np.load(pi); A=zi['re'].astype(np.float32)+1j*zi['im'].astype(np.float32)
    a=len(M); b=len(L)
    xr=start(XL.get(str(num),{}).get('rad')); xbio=start(XL.get(str(num),{}).get('bio'))
    if xr is not None and xbio is not None:
        run_cut(A,xb,fsb,xr+SCRIPT_APNEA,xbio+SCRIPT_APNEA,'마커',s,M)
    if o['grade']!='실패':
        run_cut(A,xb,fsb,o['ap'],o['ap']+o['off'],'지형지물',s,L)
    em=[x['err'] for x in M[a:]]; el=[x['err'] for x in L[b:]]
    rows.append((s,XL.get(str(num),{}).get('rad'),
                 round(xr+SCRIPT_APNEA,1) if xr is not None else None,
                 round(o['ap'],1) if o['grade']!='실패' else None,
                 len(em), np.mean(em) if em else np.nan,
                 len(el), np.mean(el) if el else np.nan))
print('%-10s %16s %10s %10s %6s %8s %6s %8s'%('피험자','파일 실험2(레이더)','마커무호흡','검출무호흡','창(마)','MAE(마)','창(지)','MAE(지)'))
for s,rg,am,al,nm_,mm,nl,ml in rows:
    print('%-10s %16s %10s %10s %6d %8s %6d %8s'%(s,str(rg)[:16],
        '%.1f'%am if am else '-','%.1f'%al if al else '-',
        nm_,'%.2f'%mm if mm==mm else '-',nl,'%.2f'%ml if ml==ml else '-'))
EM=np.array([x['err'] for x in M]); EL=np.array([x['err'] for x in L])
print('\n%-14s %6s %8s %10s %10s'%('조건','피험자','창','MAE','1BPM 이내'))
print('%-14s %6d %8d %10.2f %9.0f%%'%('마커 기반',len(set(x['s'] for x in M)),len(EM),EM.mean(),100*np.mean(EM<=1)))
print('%-14s %6d %8d %10.2f %9.0f%%'%('지형지물 기반',len(set(x['s'] for x in L)),len(EL),EL.mean(),100*np.mean(EL<=1)))
json.dump({'marker':M,'landmark':L},open('ab2.json','w'),ensure_ascii=False)
