# -*- coding: utf-8 -*-
"""싱크가 몇 초 어긋나면 호흡수 비교가 얼마나 오염되는가 — 일부러 밀어보고 측정"""
import numpy as np, json, os
from scipy.signal import butter, filtfilt, welch
W='_work'; V=''
FSR=10.0; WIN=30.0
SY={o['s']:o for o in json.load(open(V+'sync12_final.json'))}
D=np.load(V+'ds2.npz',allow_pickle=True)
SU=D['subj']; BLKn=D['blk']; TT=D['t']; BASE=D['base']; Y0=D['y_rate']
BLK={'평소 호흡':-140,'느린 호흡':-78,'회복 호흡':16,'운동 후 호흡':172}
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def rate(x,fs):
    f,P=welch(x,fs=fs,nperseg=min(len(x),int(WIN*fs)),nfft=1<<14)
    m=(f>0.06)&(f<0.80); f,P=f[m],P[m]; i=int(np.argmax(P))
    if 0<i<len(P)-1:
        a,b,c=np.log(P[i-1]+1e-30),np.log(P[i]+1e-30),np.log(P[i+1]+1e-30)
        d=np.clip(0.5*(a-c)/(a-2*b+c+1e-30),-1,1); return float((f[i]+d*(f[1]-f[0]))*60)
    return float(f[i]*60)
SHIFTS=[0,2,5,10,20]
res={s:np.full(len(Y0),np.nan) for s in SHIFTS}
cache={}
for i in range(len(Y0)):
    s=SU[i]
    if s not in cache:
        p=os.path.join(W,s+'_mot.npz')
        if not os.path.exists(p): continue
        z=np.load(p); cache[s]=(bp(z['rsp'].astype(float),0.06,0.80,float(z['fsb'])),float(z['fsb']))
    xb,fsb=cache[s]
    apb=SY[s]['ap']+SY[s]['off']
    t0=apb+float(TT[i])
    for sh in SHIFTS:
        j0,j1=int((t0+sh)*fsb),int((t0+sh+WIN)*fsb)
        if j0<0 or j1>len(xb): continue
        res[sh][i]=rate(xb[j0:j1],fsb)
print('BIOPAC 창을 일부러 밀었을 때 — 레이더(고전) 추정과의 오차 변화\n')
print('%8s %10s %12s'%('밀림','MAE','1회/분 이내'))
for sh in SHIFTS:
    m=~np.isnan(res[sh]); e=np.abs(BASE[m]-res[sh][m])
    print('%7d초 %10.2f %11.0f%%'%(sh,e.mean(),100*np.mean(e<=1)))
print('\n같은 창에서 BIOPAC 호흡수 자체가 얼마나 변하는가 (0초 대비)')
for sh in SHIFTS[1:]:
    m=(~np.isnan(res[sh]))&(~np.isnan(res[0]))
    d=np.abs(res[sh][m]-res[0][m])
    print('  %2d초 밀면  중앙값 %.2f 회/분  변화 (1회/분 이상 %d%%)'%(sh,np.median(d),int(100*np.mean(d>1))))
