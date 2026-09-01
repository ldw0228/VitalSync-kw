# -*- coding: utf-8 -*-
"""잘 맞는 파형 / 안 맞는 파형 — 레이더 복원 vs BIOPAC"""
import numpy as np, json, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.signal import butter, filtfilt
for c in ['Noto Sans CJK KR','Noto Sans CJK JP','Noto Sans CJK SC']:
    if any(c==f.name for f in fm.fontManager.ttflist): plt.rcParams['font.family']=c; break
plt.rcParams['axes.unicode_minus']=False
SURF='#fcfcfb'; INK='#0b0b0b'; SEC='#52514e'; MUT='#9aa0a6'; S2='#eb6834'; S3='#1baf7a'
W='_work'; V=''
SY={o['s']:o for o in json.load(open(V+'sync12_final.json'))}
PS=json.load(open(V+'per_subject.json'))
FSR=10.0; PAD=8.0; LO,HI=0.12,0.5
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def nz(x): return np.clip(x/(2.2*np.std(x)+1e-12),-1.4,1.4)
def pair(subj,a0,a1,rad):
    o=SY[subj]; apr=o['ap']; apb=apr+o['off']
    z=np.load(os.path.join(W,subj+'_mot.npz')); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    i0,i1=int((apb+a0-PAD)*fsb),int((apb+a1+PAD)*fsb)
    xb=bp(rsp[i0:i1],LO,HI,fsb)[int(PAD*fsb):-int(PAD*fsb)]; tb=np.arange(len(xb))/fsb+a0
    zi=np.load(os.path.join(W,subj+'_iq.npz'))
    A=zi['re'].astype(np.float32)+1j*zi['im'].astype(np.float32)
    j0,j1=int((apr+a0-PAD)*FSR),int((apr+a1+PAD)*FSR)
    s=A[rad,j0:j1].copy(); s=s-s.mean(0,keepdims=True)
    e=bp(np.abs(s),0.1,0.6,FSR).std(0); k=int(np.argmax(e))
    c=s[:,max(k-1,0):k+2].mean(1); xy=np.stack([c.real,c.imag],1); xy-=xy.mean(0)
    _,_,vt=np.linalg.svd(xy,full_matrices=False)
    w=bp(xy@vt[0],LO,HI,FSR)[int(PAD*FSR):-int(PAD*FSR)]; tr=np.arange(len(w))/FSR+a0
    if np.corrcoef(w,np.interp(tr,tb,xb))[0,1]<0: w=-w
    best=(-9,0)
    for L in np.arange(-8,8.01,.1):
        v=np.corrcoef(np.interp(tr+L,tb,xb),w)[0,1]
        if v>best[0]: best=(v,L)
    cor,lag=best
    return tb,nz(xb),tr+lag,nz(w),cor
order=sorted(PS,key=lambda k:PS[k])
GOOD=order[:3]; BAD=order[-3:][::-1]
fig,axes=plt.subplots(3,2,figsize=(11.6,7.0),dpi=200,
                      gridspec_kw=dict(hspace=.85,wspace=.16,left=.065,right=.985,top=.815,bottom=.075))
fig.patch.set_facecolor(SURF)
for col,(group,color,title) in enumerate([(GOOD,S3,'잘 맞는 사례'),(BAD,S2,'안 맞는 사례')]):
    for row,s in enumerate(group):
        ax=axes[row,col]; ax.set_facecolor(SURF)
        best=None
        for r in range(3):
            try: tb,xb,tr,w,cor=pair(s,-140,-110,r)
            except Exception: continue
            if best is None or cor>best[4]: best=(tb,xb,tr,w,cor,r)
        if best is None: continue
        tb,xb,tr,w,cor,r=best
        ax.plot(tb,xb,color=MUT,lw=1.7); ax.plot(tr,w,color=color,lw=2.0)
        ax.set_xlim(-140,-110); ax.set_ylim(-1.6,1.6)
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        for sp in ['left','bottom']: ax.spines[sp].set_color('#dfe3e8')
        ax.tick_params(colors=SEC,labelsize=8); ax.yaxis.grid(True,color='#eef1f3',lw=.7)
        ax.set_axisbelow(True)
        ax.text(0,1.10,'%s — 호흡수 오차 %.2f 회/분'%(s,PS[s]),transform=ax.transAxes,
                fontsize=10,color=INK,va='bottom',fontweight='bold')
        ax.text(1.0,1.10,'레이더 %d · 파형 상관 %.2f'%(r+1,cor),transform=ax.transAxes,
                fontsize=8.5,color=SEC,va='bottom',ha='right')
        if row==2: ax.set_xlabel('무호흡 중심 기준 시간 (초)',fontsize=9,color=SEC)
        if col==0: ax.set_ylabel('정규화 진폭',fontsize=9,color=SEC)
fig.text(.065,.945,'레이더로 복원한 호흡 파형 — 잘 되는 경우와 안 되는 경우',
         fontsize=15,color=INK,fontweight='bold')
fig.text(.065,.905,'회색 = BIOPAC 정답 · 색 = 레이더 복원 · 평소 호흡 구간 30초 · 3대 중 상관이 가장 높은 레이더\n오른쪽도 파형 자체는 꽤 따라갑니다(상관 0.62~0.86). 실패는 파형 복원이 아니라 호흡수 추출 단계에서 일어납니다',
         fontsize=9.5,color=SEC)
fig.text(.065,.873,'왼쪽 = 호흡수가 잘 맞는 3명',fontsize=10,color=S3,fontweight='bold')
fig.text(.545,.873,'오른쪽 = 호흡수가 안 맞는 3명',fontsize=10,color=S2,fontweight='bold')
fig.savefig('fig_waves.png',facecolor=SURF)
print('ok', GOOD, BAD)
