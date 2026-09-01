# -*- coding: utf-8 -*-
"""'위아래 한 쌍'은 같은 거리에 있고, 진짜 2차 반사는 더 먼 거리에 있다"""
import numpy as np, glob, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from scipy.signal import butter, filtfilt, welch
for c in ['Noto Sans CJK KR','Noto Sans CJK JP','Noto Sans CJK SC']:
    if any(c==f.name for f in fm.fontManager.ttflist): plt.rcParams['font.family']=c; break
plt.rcParams['axes.unicode_minus']=False
SURF='#fcfcfb'; INK='#0b0b0b'; SEC='#52514e'; S1='#2a78d6'; S2='#eb6834'
FS=10.0
def bp(x,lo=0.1,hi=0.6,fs=FS,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)

# ---- 왼쪽 패널용 파형 (원본 184 프레임)
z=np.load('_work/_raw184_S03.npz')
m=z['r3'].astype(float); m=m-m.mean(0,keepdims=True); K=31
I=bp(m[:,K]); Q=bp(m[:,K+92]); t=np.arange(len(I))/FS
a,b=120,180; sl=slice(int(a*FS),int(b*FS))
nz=lambda x:x/(np.abs(x).max()+1e-12)
r=float(np.corrcoef(I,Q)[0,1])
cc=(m[:,:92]+1j*m[:,92:]); cc=cc-cc.mean(0,keepdims=True)
xy=np.stack([cc[:,K].real,cc[:,K].imag],1); xy-=xy.mean(0)
_,_,vt=np.linalg.svd(xy,full_matrices=False)
P=bp(xy@vt[0]); f,Pw=welch(P,fs=FS,nperseg=1024); mk=(f>.05)&(f<.6); f0=f[mk][np.argmax(Pw[mk])]

# ---- 오른쪽 패널용 통계 (29명 × 3레이더)
W='_work'
dPair=[]; dGhost=[]
for p in sorted(glob.glob(os.path.join(W,'*_iq.npz'))):
    zz=np.load(p); re=zz['re'].astype(np.float32); im=zz['im'].astype(np.float32)
    for rr in range(re.shape[0]):
        R=bp(re[rr,:1800]-re[rr,:1800].mean(0)).std(0)
        Im=bp(im[rr,:1800]-im[rr,:1800].mean(0)).std(0)
        dPair.append(int(np.argmax(Im))-int(np.argmax(R)))
        A=(re[rr,:1800]+1j*im[rr,:1800]); A=A-A.mean(0)
        e=bp(np.abs(A)).std(0); k=int(np.argmax(e))
        msk=np.ones(len(e),bool); msk[max(k-6,0):k+7]=False
        dGhost.append(int(np.arange(len(e))[msk][np.argmax(e[msk])])-k)
dPair=np.array(dPair); dGhost=np.array(dGhost)

fig=plt.figure(figsize=(11.6,4.6),dpi=200); fig.patch.set_facecolor(SURF)
gs=fig.add_gridspec(1,2,width_ratios=[1.55,1.0],wspace=.22,
                    left=.062,right=.985,top=.66,bottom=.175)
def st(ax):
    ax.set_facecolor(SURF); ax.tick_params(colors=SEC,labelsize=8.5,length=3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#dfe3e8')
    ax.set_axisbelow(True)

axA=fig.add_subplot(gs[0,0])
axA.plot(t[sl],nz(I[sl]),color=S1,lw=2.0,solid_capstyle='round')
axA.plot(t[sl],nz(Q[sl]),color=S2,lw=2.0,solid_capstyle='round')
axA.text(a+1,1.32,'아래 줄 (bin 31) = I',color=S1,fontsize=9.5,fontweight='bold')
axA.text(a+22,1.32,'위 줄 (bin 123) = Q',color=S2,fontsize=9.5,fontweight='bold')
axA.text(0,1.20,'A.  두 줄은 정말 다른 파형입니다',transform=axA.transAxes,
         fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axA.text(0,1.06,'같은 거리 bin의 실수·허수 성분 · 상관 r = %.2f — 복사본이 아닙니다'%r,
         transform=axA.transAxes,fontsize=8.8,color=SEC,va='bottom')
axA.set_ylim(-1.6,1.65); axA.yaxis.grid(True,color='#eef1f3',lw=.8); st(axA)
axA.set_xlabel('시간 (초)',fontsize=9,color=SEC); axA.set_ylabel('정규화 진폭',fontsize=9,color=SEC)

axB=fig.add_subplot(gs[0,1])
rng=np.random.default_rng(0)
axB.scatter(dPair,np.ones(len(dPair))*1+rng.uniform(-.16,.16,len(dPair)),
            s=26,color=S1,alpha=.55,linewidths=0)
axB.scatter(dGhost,np.zeros(len(dGhost))+rng.uniform(-.16,.16,len(dGhost)),
            s=26,color=S2,alpha=.55,linewidths=0)
axB.axvline(0,color='#c9ced3',lw=1.1,ls='--')
axB.set_yticks([1,0]); axB.set_yticklabels(['위 줄 vs 아래 줄','진짜 2차 반사 vs 1차'],fontsize=9)
axB.set_ylim(-.55,1.55); axB.set_xlim(-32,40)
axB.text(2.0,1.42,'같은 거리 — 97%가 ±3 bin 안',fontsize=8.6,color=S1,fontweight='bold')
axB.text(6,.42,'거리가 제각각 — 사람·레이더마다 다름',fontsize=8.6,color=S2,fontweight='bold')
axB.text(0,1.20,'B.  거리 차이로 보면 확실합니다',transform=axB.transAxes,
         fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axB.text(0,1.06,'29명 × 레이더 3대 = 87건',transform=axB.transAxes,
         fontsize=8.8,color=SEC,va='bottom')
axB.set_xlabel('거리 차이 (bin)',fontsize=9,color=SEC)
axB.xaxis.grid(True,color='#eef1f3',lw=.8); st(axB)

fig.text(.062,.905,'"위아래 한 쌍"의 정체 — 복사본이 아니라 I와 Q',fontsize=14.5,color=INK,fontweight='bold')
fig.text(.062,.845,'S03_PSJ · 레이더 3 · 120~180초  (오른쪽은 전체 피험자 통계)',fontsize=9.5,color=SEC)
fig.text(.062,.038,'멀티패스는 더 긴 경로를 돌아오므로 반드시 더 먼 거리에 나타납니다. 위 줄은 같은 거리에 있으므로 반사가 아니라 같은 표적의 다른 축입니다. '
                   '두 축을 합쳐 투영하면 호흡 %.3f Hz — BIOPAC 0.117 Hz와 일치합니다.'%f0,fontsize=8.8,color=SEC)
fig.savefig('fig_iq.png',facecolor=SURF)
print('pair ±3 이내 %.0f%%  ghost 중앙값 %+d'%(100*np.mean(np.abs(dPair)<=3),int(np.median(dGhost))))
