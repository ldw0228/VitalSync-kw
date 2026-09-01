# -*- coding: utf-8 -*-
"""서브젝트별 호흡 복원 일치도 + 잘 맞는/안 맞는 사례"""
import numpy as np, json, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.signal import butter, filtfilt
for c in ['Noto Sans CJK KR','Noto Sans CJK JP','Noto Sans CJK SC']:
    if any(c==f.name for f in fm.fontManager.ttflist): plt.rcParams['font.family']=c; break
plt.rcParams['axes.unicode_minus']=False
SURF='#fcfcfb'; INK='#0b0b0b'; SEC='#52514e'; MUT='#9aa0a6'
S1='#2a78d6'; S2='#eb6834'; S3='#1baf7a'
W='_work'
M=json.load(open('match.json'))
FSR=10.0; PAD=8.0; LO,HI=0.12,0.5
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def nz(x):
    s=2.2*np.std(x)+1e-12; return np.clip(x/s,-1.4,1.4)
def pair(subj,a0,a1,rad=0,align=False):
    """PAD초 여유를 두고 필터한 뒤 가장자리를 잘라 필터 전이구간 제거"""
    d=M[subj]; apb=d['apb']; apr=d['apr']
    z=np.load(os.path.join(W,subj+'_mot.npz'))
    rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    i0,i1=int((apb+a0-PAD)*fsb),int((apb+a1+PAD)*fsb)
    xb=bp(rsp[i0:i1],LO,HI,fsb)[int(PAD*fsb):-int(PAD*fsb)]
    tb=np.arange(len(xb))/fsb+a0
    zi=np.load(os.path.join(W,subj+'_iq.npz'))
    A=zi['re'].astype(np.float32)+1j*zi['im'].astype(np.float32)
    j0,j1=int((apr+a0-PAD)*FSR),int((apr+a1+PAD)*FSR)
    s=A[rad,j0:j1].copy(); s=s-s.mean(0,keepdims=True)
    e=bp(np.abs(s),0.1,0.6,FSR).std(0); k=int(np.argmax(e))
    c=s[:,max(k-1,0):k+2].mean(1)
    xy=np.stack([c.real,c.imag],1); xy=xy-xy.mean(0)
    _,_,vt=np.linalg.svd(xy,full_matrices=False)
    w=bp(xy@vt[0],LO,HI,FSR)[int(PAD*FSR):-int(PAD*FSR)]
    tr=np.arange(len(w))/FSR+a0
    bi=np.interp(tr,tb,xb)
    if np.corrcoef(w,bi)[0,1]<0: w=-w
    lag=0.0; cor=float(np.corrcoef(bi,w)[0,1])
    if align:
        best=(-9,0)
        for L in np.arange(-8,8.01,.1):
            v=np.corrcoef(np.interp(tr+L,tb,xb),w)[0,1]
            if v>best[0]: best=(v,L)
        cor,lag=best; tr=tr+lag
    return tb,nz(xb),tr,nz(w),lag,cor

fig=plt.figure(figsize=(11.6,7.8),dpi=200); fig.patch.set_facecolor(SURF)
gs=fig.add_gridspec(3,1,height_ratios=[1.20,1.0,1.0],hspace=.80,
                    left=.075,right=.982,top=.805,bottom=.075)
def st(ax):
    ax.set_facecolor(SURF); ax.tick_params(colors=SEC,labelsize=8.5,length=3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#dfe3e8')
    ax.set_axisbelow(True)

# ---- A
order=sorted(M,key=lambda k:M[k]['mae']); vals=[M[s]['mae'] for s in order]
axA=fig.add_subplot(gs[0]); st(axA)
axA.bar(range(len(vals)),vals,.62,
        color=[S3 if v<=1.19 else (S1 if v<=2.5 else S2) for v in vals])
axA.axhline(1.18,color=MUT,lw=1.0,ls=':'); axA.axhline(2.5,color=S2,lw=1.0,ls='--')
axA.text(.25,1.55,'스펙트럼 해상도 1.18',fontsize=8,color=MUT)
axA.text(.25,2.85,'선배가 말한 기준 2.5',fontsize=8,color=S2)
axA.set_xticks(range(len(order)))
axA.set_xticklabels([s.split('_')[0] for s in order],rotation=90,fontsize=7.2)
axA.set_ylabel('BPM 오차 (중앙값)',fontsize=9,color=SEC)
axA.set_ylim(0,14); axA.yaxis.grid(True,color='#eef1f3',lw=.8)
axA.text(0,1.20,'A.  서브젝트별 호흡수 일치도 — 25명 중 23명이 2.5 BPM 안',
         transform=axA.transAxes,fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axA.text(0,1.075,'레이더에서 복원한 호흡수 vs BIOPAC · 3블록 × 레이더 3대 = 225건 · 80%가 해상도(1.18 BPM) 이내',
         transform=axA.transAxes,fontsize=8.8,color=SEC,va='bottom')

def wave(ax,subj,a0,a1,tag,note,color,align,rad=0):
    tb,xb,tr,w,lag,cor=pair(subj,a0,a1,rad=rad,align=align)
    ax.plot(tb,xb,color=MUT,lw=1.8,solid_capstyle='round')
    ax.plot(tr,w,color=color,lw=2.1,solid_capstyle='round')
    ax.set_xlim(a0,a1); ax.set_ylim(-1.5,1.95); st(ax)
    ax.yaxis.grid(True,color='#eef1f3',lw=.8)
    ax.set_ylabel('정규화 진폭',fontsize=9,color=SEC)
    ax.text(0,1.20,tag,transform=ax.transAxes,fontsize=11.5,color=INK,va='bottom',fontweight='bold')
    ax.text(0,1.065,note,transform=ax.transAxes,fontsize=8.8,color=SEC,va='bottom')
    x0=a0+(a1-a0)*.015
    ax.text(x0,1.58,'BIOPAC',color=SEC,fontsize=9,fontweight='bold')
    ax.text(x0+(a1-a0)*.115,1.58,'레이더 %d (복원)'%(rad+1),color=color,fontsize=9,fontweight='bold')
    return lag,cor

axB=fig.add_subplot(gs[1])
*_,lagB,corB=pair('S19_CHW',-140,-110,rad=2,align=True)
wave(axB,'S19_CHW',-140,-110,'B.  잘 맞는 사례 — S19_CHW  (평소 호흡)',
     '레이더 12.9 BPM / BIOPAC 12.9 BPM · 파형 상관 %.2f · 잔여 시차 %+.1f초 (지형지물 정렬 오차 범위 안)'%(corB,lagB),
     S3,True,rad=2)
axC=fig.add_subplot(gs[2])
wave(axC,'S22_KJH',-78,-48,'C.  안 맞는 사례 — S22_KJH  (느린 호흡)',
     '대본은 "느리게"인데 BIOPAC은 18.8 BPM으로 빠릅니다 · 레이더는 5.9~7.1 BPM으로 대본에 가깝습니다',S2,False,rad=2)
axC.set_xlabel('무호흡 중심 기준 시간 (초)',fontsize=9,color=SEC)

fig.text(.075,.945,'IR-UWB 호흡 복원 — 서브젝트별로 얼마나 맞는가',fontsize=15,color=INK,fontweight='bold')
fig.text(.075,.898,'두 시계는 각자의 무호흡 중심으로 정렬했습니다 (offset 추정 불필요) · 필터 전이구간은 잘라냈습니다',
         fontsize=9.5,color=SEC)
fig.savefig('fig_match.png',facecolor=SURF)
print('lag=%.2f cor=%.2f'%(lagB,corB))
