# -*- coding: utf-8 -*-
"""기준선 — 28명 호흡수 일치도, 잘 맞는 사례, 그리고 레이더 선택 문제"""
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
PS=json.load(open('per_subject.json'))
BL=json.load(open('baseline3.json'))
SY={o['s']:o for o in json.load(open('sync12_final.json'))}
MM=json.load(open('match.json'))
FSR=10.0; PAD=8.0; LO,HI=0.12,0.5
def bp(x,lo,hi,fs,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def nz(x): s=2.2*np.std(x)+1e-12; return np.clip(x/s,-1.4,1.4)
def pair(subj,a0,a1,rad,align=True):
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
    if align:
        best=(-9,0)
        for L in np.arange(-8,8.01,.1):
            v=np.corrcoef(np.interp(tr+L,tb,xb),w)[0,1]
            if v>best[0]: best=(v,L)
        cor,lag=best; tr=tr+lag
    t0=max(tb[0],tr[0])+1.0; t1=min(tb[-1],tr[-1])-1.0     # 겹치는 구간만
    mb=(tb>=t0)&(tb<=t1); mr=(tr>=t0)&(tr<=t1)
    return tb[mb],nz(xb[mb]),tr[mr],nz(w[mr]),cor,t0,t1

fig=plt.figure(figsize=(11.6,8.0),dpi=200); fig.patch.set_facecolor(SURF)
gs=fig.add_gridspec(3,1,height_ratios=[1.20,0.95,1.05],hspace=.82,
                    left=.078,right=.982,top=.80,bottom=.07)
def st(ax):
    ax.set_facecolor(SURF); ax.tick_params(colors=SEC,labelsize=8.5,length=3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#dfe3e8')
    ax.set_axisbelow(True)

# A
order=sorted(PS,key=lambda k:PS[k]); vals=[PS[s] for s in order]
axA=fig.add_subplot(gs[0]); st(axA)
axA.bar(range(len(vals)),vals,.64,
        color=[S3 if v<=1.0 else (S1 if v<=2.5 else S2) for v in vals])
axA.axhline(1.0,color=MUT,lw=1.0,ls=':'); axA.axhline(2.5,color=S2,lw=1.0,ls='--')
axA.text(.25,1.15,'1 BPM',fontsize=8,color=MUT); axA.text(.25,2.68,'2.5 BPM',fontsize=8,color=S2)
axA.set_xticks(range(len(order))); axA.set_xticklabels([s.split('_')[0] for s in order],rotation=90,fontsize=7.2)
axA.set_ylabel('호흡수 오차 (BPM)',fontsize=9,color=SEC); axA.set_ylim(0,7.6)
axA.yaxis.grid(True,color='#eef1f3',lw=.8)
axA.text(0,1.20,'A.  기준선 — 28명 중 20명이 1 BPM 이내, 25명이 2.5 BPM 이내',
         transform=axA.transAxes,fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axA.text(0,1.075,'레이더 3대 중앙값 융합 · 평소·느린·회복·운동 후 4구간 평균 · 전체 MAE 1.08 BPM',
         transform=axA.transAxes,fontsize=8.8,color=SEC,va='bottom')

# B
axB=fig.add_subplot(gs[1]); 
def pairM(subj,a0,a1,rad):
    d=MM[subj]; apb=d['apb']; apr=d['apr']
    z=np.load(os.path.join(W,subj+'_mot.npz')); rsp=z['rsp'].astype(float); fsb=float(z['fsb'])
    i0,i1=int((apb+a0-PAD)*fsb),int((apb+a1+PAD)*fsb)
    xb=bp(rsp[i0:i1],LO,HI,fsb)[int(PAD*fsb):-int(PAD*fsb)]; tb=np.arange(len(xb))/fsb+a0
    zi=np.load(os.path.join(W,subj+'_iq.npz'))
    A=zi['re'].astype(np.float32)+1j*zi['im'].astype(np.float32)
    j0,j1=int((apr+a0-PAD)*FSR),int((apr+a1+PAD)*FSR)
    s_=A[rad,j0:j1].copy(); s_=s_-s_.mean(0,keepdims=True)
    e=bp(np.abs(s_),0.1,0.6,FSR).std(0); k=int(np.argmax(e))
    c=s_[:,max(k-1,0):k+2].mean(1); xy=np.stack([c.real,c.imag],1); xy-=xy.mean(0)
    _,_,vt=np.linalg.svd(xy,full_matrices=False)
    w=bp(xy@vt[0],LO,HI,FSR)[int(PAD*FSR):-int(PAD*FSR)]; tr=np.arange(len(w))/FSR+a0
    if np.corrcoef(w,np.interp(tr,tb,xb))[0,1]<0: w=-w
    best=(-9,0)
    for L in np.arange(-8,8.01,.1):
        v=np.corrcoef(np.interp(tr+L,tb,xb),w)[0,1]
        if v>best[0]: best=(v,L)
    cor,lag=best; tr=tr+lag
    return tb,nz(xb),tr,nz(w),cor
tb,xb,tr,w,cor=pairM('S19_CHW',-140,-110,2); t0,t1=-140,-110
axB.plot(tb,xb,color=MUT,lw=1.8); axB.plot(tr,w,color=S3,lw=2.1)
axB.set_xlim(t0,t1); axB.set_ylim(-1.5,1.95); st(axB)
axB.yaxis.grid(True,color='#eef1f3',lw=.8)
axB.set_ylabel('정규화 진폭',fontsize=9,color=SEC)
axB.set_xlabel('무호흡 중심 기준 시간 (초)',fontsize=9,color=SEC)
axB.text(0,1.22,'B.  잘 맞는 사례 — S19_CHW (평균 오차 0.10 BPM)',transform=axB.transAxes,
         fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axB.text(0,1.08,'네 구간 모두 일치 — 12.9/12.9 · 6.1/6.0 · 22.4/22.6 · 30.6/30.6 BPM (BIOPAC/레이더)',
         transform=axB.transAxes,fontsize=8.8,color=SEC,va='bottom')
axB.text(t0+0.4,1.58,'BIOPAC',color=SEC,fontsize=9,fontweight='bold')
axB.text(t0+4.0,1.58,'레이더 3 (복원)',color=S3,fontsize=9,fontweight='bold')

# C — 레이더 선택 문제
axC=fig.add_subplot(gs[2]); st(axC)
S='S22_KJH'
blks=['평소 호흡','느린 호흡','회복 호흡','운동 후 호흡']
rows=[r for b in blks for r in BL if r['s']==S and r['blk']==b]
cols=[S1,S2,S3]
for i,r in enumerate(rows):
    y=len(rows)-1-i
    bio=r['bio']['스펙트럼']; rad=r['rad']['스펙트럼']; med=float(np.median(rad))
    axC.plot([0,34],[y,y],color='#f0f2f4',lw=8,solid_capstyle='round',zorder=0)
    axC.plot([bio,bio],[y-.30,y+.30],color=INK,lw=2.4,zorder=4)
    for k in range(3):
        axC.plot([rad[k]],[y],'o',ms=9,color=cols[k],zorder=3,
                 markeredgecolor=SURF,markeredgewidth=1.4)
    axC.plot([med],[y],'o',ms=15,mfc='none',mec=INK,mew=1.8,zorder=5)
    ok=min(range(3),key=lambda k:abs(rad[k]-bio))
    axC.text(34.6,y,'정답에 가장 가까운 건 레이더 %d (%.1f)'%(ok+1,rad[ok]),
             fontsize=8.4,color=SEC,va='center')
axC.set_yticks(range(len(rows))); axC.set_yticklabels(blks[::-1],fontsize=9)
axC.set_xlim(0,48); axC.set_ylim(-.6,len(rows)-.4)
axC.set_xticks([0,10,20,30]); axC.set_xlabel('호흡수 (BPM)',fontsize=9,color=SEC)
axC.xaxis.grid(True,color='#eef1f3',lw=.8)
axC.text(0,1.19,'C.  안 맞는 사례 — S22_KJH (6.96 BPM). 정보는 있는데 고르지 못합니다',
         transform=axC.transAxes,fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axC.text(0,1.06,'세로 막대 = BIOPAC 정답 · 색 점 = 레이더 1·2·3 · 빈 원 = 3대 중앙값(현재 방식)',
         transform=axC.transAxes,fontsize=8.8,color=SEC,va='bottom')

fig.text(.078,.945,'호흡수 복원 기준선 — 모델이 이겨야 할 숫자',fontsize=15,color=INK,fontweight='bold')
fig.text(.078,.898,'실험 1·2 정박 28명 · 레이더 3대 중앙값 융합 MAE 1.08 BPM · 매 순간 최선의 레이더를 고를 수 있다면 0.32 BPM',
         fontsize=9.5,color=SEC)
fig.savefig('fig_base.png',facecolor=SURF)
print('ok')
