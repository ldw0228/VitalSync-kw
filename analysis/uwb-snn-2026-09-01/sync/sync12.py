# -*- coding: utf-8 -*-
"""실험 1·2 만으로 한정했을 때 동기화가 되는 인원 — 운동 구간 탐색에 의존하지 않음"""
import numpy as np, json, os, glob
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d
W='_work'
LM=json.load(open('landmarks.json')); FSR=10.0
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def dip(env,fs,t0,lo,hi,win=20.0):
    """[lo,hi]초 안에서 win초 이동평균 포락선이 가장 낮은 지점"""
    k=int(win*fs); ma=uniform_filter1d(env,k)
    i0=max(int((lo-t0)*fs),k//2); i1=min(int((hi-t0)*fs),len(ma)-k//2)
    if i1-i0<k: return None
    i=int(np.argmin(ma[i0:i1]))+i0
    return t0+i/fs, float(ma[i]/(np.median(ma)+1e-12))
rows=[]
for p in sorted(glob.glob(os.path.join(W,'*_iq.npz'))):
    s=os.path.basename(p)[:-7]
    pm=os.path.join(W,s+'_mot.npz')
    if s not in LM or not os.path.exists(pm):
        rows.append((s,None,None,None,None,'실험1 회전 미검출')); continue
    end1=LM[s]['blocks'][2][1]
    lo,hi=end1+100,end1+260
    # --- 레이더: 3대 평균 호흡 포락선
    z=np.load(p); A=z['re'].astype(np.float32)+1j*z['im'].astype(np.float32)
    envs=[]
    for r in range(A.shape[0]):
        j0,j1=0,A.shape[1]
        seg=A[r,j0:j1]-A[r,j0:j1].mean(0,keepdims=True)
        e=bp(np.abs(seg),0.1,0.6,FSR).std(0); kb=int(np.argmax(e))
        c=seg[:,max(kb-1,0):kb+2].mean(1)
        xy=np.stack([c.real,c.imag],1); xy-=xy.mean(0)
        _,_,vt=np.linalg.svd(xy,full_matrices=False)
        envs.append(uniform_filter1d(np.abs(bp(xy@vt[0],0.1,0.6,FSR)),int(4*FSR)))
    R=dip(np.mean(envs,0),FSR,0,lo,hi)
    # --- BIOPAC
    zm=np.load(pm); rsp=zm['rsp'].astype(float); fsb=float(zm['fsb'])
    xb=bp(rsp,0.08,0.8,fsb); eb=uniform_filter1d(np.abs(xb),int(4*fsb))
    Bp=dip(eb,fsb,0,lo-30,hi+30)
    if R is None or Bp is None:
        rows.append((s,None,None,None,None,'구간 부족')); continue
    apr,dr=R; apb,db=Bp; off=apb-apr
    ok = dr<0.5 and db<0.5 and -20<=off<=15
    why='' if ok else ('무호흡 함몰 약함' if (dr>=0.5 or db>=0.5) else 'offset 비정상 %.1f초'%off)
    rows.append((s,apr,dr,off,db,'통과' if ok else why))
print('%-10s %8s %7s %8s %7s  %s'%('피험자','레이더무호흡','함몰','offset','BIO함몰','판정'))
ok=0
for s,apr,dr,off,db,st in rows:
    if apr is None: print('%-10s %8s %7s %8s %7s  %s'%(s,'-','-','-','-',st)); continue
    if st=='통과': ok+=1
    print('%-10s %8.1f %7.2f %+8.1f %7.2f  %s'%(s,apr,dr,off,db,st))
print('\n통과 %d / %d명'%(ok,len(rows)))
json.dump([{'s':r[0],'ap_radar':r[1],'depth':r[2],'offset':r[3],'bio_depth':r[4],'status':r[5]} for r in rows],
          open('sync12.json','w'),ensure_ascii=False,indent=1)
