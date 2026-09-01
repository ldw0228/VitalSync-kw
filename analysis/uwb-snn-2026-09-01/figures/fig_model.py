# -*- coding: utf-8 -*-
"""SNN 후보 선택기 결과"""
import numpy as np, json, os, glob, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for c in ['Noto Sans CJK KR','Noto Sans CJK JP','Noto Sans CJK SC']:
    if any(c==f.name for f in fm.fontManager.ttflist): plt.rcParams['font.family']=c; break
plt.rcParams['axes.unicode_minus']=False
SURF='#fcfcfb'; INK='#0b0b0b'; SEC='#52514e'; MUT='#9aa0a6'
S1='#2a78d6'; S2='#eb6834'; S3='#1baf7a'
V=''
D=np.load(V+'ds_rate.npz',allow_pickle=True)
EST=D['est']; Y=D['y']; ERR=D['err']; BLK=D['blk']; SUBJ=D['subj']
SUB=sorted(set(SUBJ.tolist())); FOLD=[SUB[i::4] for i in range(4)]
def load(enc):
    p=V+'sel_%s.json'%enc
    if not os.path.exists(p): return None
    res=json.load(open(p))
    if len(res)<4: return None
    pick=np.zeros(len(Y))
    for f,r in enumerate(res): pick[np.isin(SUBJ,FOLD[f])]=np.array(r['pick'])
    return pick,res
base=np.abs(np.median(EST[:,[0,3,6]],axis=1)-Y); orac=ERR.min(1)
rate=load('rate'); pick=rate[0]
ENCS=[('rate','rate'),('sf','step-forward'),('pop','population'),('delta','delta'),('direct','direct')]
have=[(k,n,load(k)) for k,n in ENCS]; have=[(k,n,v) for k,n,v in have if v]

nrow=2 if len(have)>1 else 1
fig=plt.figure(figsize=(11.6,7.4 if nrow==2 else 4.2),dpi=200); fig.patch.set_facecolor(SURF)
gs=fig.add_gridspec(1,2,wspace=.24,left=.075,right=.982,
                    top=.80 if nrow==2 else .70,bottom=.53 if nrow==2 else .09)
def st(ax):
    ax.set_facecolor(SURF); ax.tick_params(colors=SEC,labelsize=8.5,length=3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#dfe3e8')
    ax.set_axisbelow(True)

# A. 구간별 기준선 → SNN → 상한
axA=fig.add_subplot(gs[0,0]); st(axA)
BL=['평소 호흡','느린 호흡','회복 호흡','운동 후 호흡']
x=np.arange(len(BL)); w=.26
b=[base[BLK==k].mean() for k in BL]; s=[pick[BLK==k].mean() for k in BL]; o=[ERR[BLK==k].min(1).mean() for k in BL]
axA.bar(x-w,b,w,color=MUT,label='기준선 (중앙값 융합)')
axA.bar(x,s,w,color=S1,label='SNN 선택기')
axA.bar(x+w,o,w,color=S3,label='상한 (최선 후보)')
axA.set_xticks(x); axA.set_xticklabels(BL,fontsize=8.5)
axA.set_ylabel('호흡수 오차 MAE (BPM)',fontsize=9,color=SEC)
axA.yaxis.grid(True,color='#eef1f3',lw=.8); axA.legend(fontsize=8,frameon=False)
axA.text(0,1.16,'A.  구간별 성능',transform=axA.transAxes,fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axA.text(0,1.045,'느린 호흡에서 가장 크게 좋아지고, 운동 후는 거의 그대로입니다',
         transform=axA.transAxes,fontsize=8.8,color=SEC,va='bottom')

# B. 오차 누적분포
axB=fig.add_subplot(gs[0,1]); st(axB)
for e,c,l,lw,px,dy in [(base,MUT,'기준선',2.0,.92,-9),(pick,S1,'SNN 선택기',2.4,.80,-9),(orac,S3,'상한',2.0,.55,+5)]:
    xs=np.sort(e); ys=np.arange(1,len(xs)+1)/len(xs)*100
    axB.plot(xs,ys,color=c,lw=lw,solid_capstyle='round')
    i=int(len(xs)*px); axB.text(min(xs[i],3.6),ys[i]+dy,l,color=c,fontsize=9,fontweight='bold',ha='center')
axB.axvline(1.0,color='#dfe3e8',lw=1.0,ls=':')
axB.text(1.1,12,'1 BPM',fontsize=8,color=MUT)
axB.set_xlim(0,4); axB.set_ylim(0,101)
axB.set_xlabel('호흡수 오차 (BPM)',fontsize=9,color=SEC)
axB.set_ylabel('누적 비율 (%)',fontsize=9,color=SEC)
axB.yaxis.grid(True,color='#eef1f3',lw=.8)
axB.text(0,1.16,'B.  오차 분포 — 784개 창',transform=axB.transAxes,fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axB.text(0,1.045,'1 BPM 이내 비율 79%% → %d%% (상한 %d%%)'%(100*np.mean(pick<=1),100*np.mean(orac<=1)),
         transform=axB.transAxes,fontsize=8.8,color=SEC,va='bottom')

if nrow==2:
    axC=fig.add_axes([.245,.075,.735,.31]); st(axC)
    ab=json.load(open(V+'ablate.json'))
    jo=json.load(open(V+'joint.json'))
    rows=[('파형만 · 9개 공동 채점',jo['mae'],100*jo['p1'],S2),
          ('파형만 · 후보별 채점',ab['snn']['mae'],100*ab['snn']['p1'],S2),
          ('기준선 · 3대 중앙값 (학습 없음)',base.mean(),100*np.mean(base<=1),MUT),
          ('신뢰도 특징만 (SNN 없음)',ab['aux']['mae'],100*ab['aux']['p1'],'#7f9fb0'),
          ('특징 + SNN (최종)',pick.mean(),100*np.mean(pick<=1),S1),
          ('상한 · 매번 최선 후보',orac.mean(),100*np.mean(orac<=1),S3)]
    rows=sorted(rows,key=lambda r:-r[1])
    y=np.arange(len(rows))
    axC.barh(y,[r[1] for r in rows],.58,color=[r[3] for r in rows])
    for i,r in enumerate(rows):
        axC.text(r[1]+.04,i,'%.2f   1 BPM 이내 %.0f%%'%(r[1],r[2]),va='center',fontsize=8.8,color=INK)
    axC.set_yticks(y); axC.set_yticklabels([r[0] for r in rows],fontsize=9)
    axC.set_xlim(0,2.9); axC.set_xlabel('호흡수 오차 MAE (BPM)',fontsize=9,color=SEC)
    axC.xaxis.grid(True,color='#eef1f3',lw=.8)
    axC.text(0,1.10,'C.  무엇이 성능을 만드는가 — 절제 실험',transform=axC.transAxes,
             fontsize=11.5,color=INK,va='bottom',fontweight='bold')
    axC.text(0,1.02,'파형만으로는 기준선을 넘지 못합니다. 이득의 대부분은 신뢰도 특징에서 나오고, SNN은 그 위에 8%를 더합니다',
             transform=axC.transAxes,fontsize=8.8,color=SEC,va='bottom')

fig.text(.075,.945,'후보 선택기 — 기준선 1.66 → 1.18 BPM, 상한은 0.46',fontsize=15,color=INK,fontweight='bold')
fig.text(.075,.90 if nrow==2 else .87,
  '9개 후보(레이더 3대 × 복조 3방식)를 스파이크로 읽고 신뢰도를 매겨 하나를 고릅니다 · 30초 창 784개 · 피험자 분리 4-fold',
  fontsize=9.5,color=SEC)
fig.savefig('fig_model.png',facecolor=SURF)
print('ok  인코딩 %d종 반영'%len(have))
