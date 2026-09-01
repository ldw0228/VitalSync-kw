# -*- coding: utf-8 -*-
"""12조합 최종 결과 (train_v2.json — 학습/검증/평가 3분할, 조기종료)"""
import numpy as np, json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for c in ['Noto Sans CJK KR','Noto Sans CJK JP','Noto Sans CJK SC']:
    if any(c==f.name for f in fm.fontManager.ttflist): plt.rcParams['font.family']=c; break
plt.rcParams['axes.unicode_minus']=False
SURF='#fcfcfb'; INK='#0b0b0b'; SEC='#52514e'; MUT='#9aa0a6'
S1='#2a78d6'; S2='#eb6834'; S3='#1baf7a'
R=json.load(open('train_v2.json'))
NAME={'-':'—','rate':'rate','sf':'step-forward','pop':'population','delta':'delta','direct':'direct'}
def st(ax):
    ax.set_facecolor(SURF); ax.tick_params(colors=SEC,labelsize=8.5,length=3)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color('#dfe3e8')
    ax.set_axisbelow(True)

fig=plt.figure(figsize=(11.6,7.6),dpi=200); fig.patch.set_facecolor(SURF)

# A. 12조합 막대 (자르기별 그룹)
axA=fig.add_axes([.20,.545,.775,.29]); st(axA)
rows=[]
for mode,lab in [('marker','마커'),('landmark','지형지물')]:
    sub=[r for r in R if r['mode']==mode]
    rows.append(('%s · 고전 신호처리'%lab, sub[0]['base_mae'], 100*sub[0]['base_p1'], 0, MUT, mode))
    for r in sub:
        nm='%s · %s'%(lab,'ANN' if r['kind']=='ANN' else 'SNN '+NAME[r['enc']])
        rows.append((nm, r['mae'], 100*r['p1'], r['params'], S2 if mode=='marker' else S1, mode))
rows=sorted(rows,key=lambda r:-r[1])
best=min(r[1] for r in rows)
y=np.arange(len(rows))
cols=[('#1baf7a' if abs(r[1]-best)<1e-9 else r[4]) for r in rows]
axA.barh(y,[r[1] for r in rows],.62,color=cols)
for i,r in enumerate(rows):
    axA.text(r[1]+.02,i,'%.2f    1BPM %.0f%%    파라미터 %s'%(r[1],r[2],'없음' if r[3]==0 else format(r[3],',')),
             va='center',fontsize=8.2,color=INK)
axA.set_yticks(y); axA.set_yticklabels([r[0] for r in rows],fontsize=8.6)
axA.set_xlim(0,2.75); axA.set_xlabel('호흡수 오차 MAE (회/분)',fontsize=9,color=SEC)
axA.xaxis.grid(True,color='#eef1f3',lw=.8)
axA.text(0,1.09,'A.  14개 조합 전체 — 자르기 2방식 × (고전 + ANN + SNN 5인코딩)',
         transform=axA.transAxes,fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axA.text(0,1.005,'주황=마커 기반, 파랑=지형지물 기반, 회색=학습 없는 고전 기준선, 초록=최고 성능',
         transform=axA.transAxes,fontsize=8.6,color=SEC,va='bottom')

# B. 자르기 효과 vs 모델 효과
axB=fig.add_axes([.075,.085,.38,.33]); st(axB)
mk=[r['mae'] for r in R if r['mode']=='marker']; lm=[r['mae'] for r in R if r['mode']=='landmark']
mkb=[r['base_mae'] for r in R if r['mode']=='marker'][0]; lmb=[r['base_mae'] for r in R if r['mode']=='landmark'][0]
axB.scatter(np.full(len(mk),0)+np.random.RandomState(0).uniform(-.06,.06,len(mk)),mk,s=44,color=S2,zorder=3,label='학습 모델')
axB.scatter(np.full(len(lm),1)+np.random.RandomState(1).uniform(-.06,.06,len(lm)),lm,s=44,color=S1,zorder=3)
axB.scatter([0,1],[mkb,lmb],s=70,marker='_',color=MUT,zorder=4,lw=2.4)
axB.text(0,mkb-.055,'고전 %.2f'%mkb,ha='center',fontsize=8,color=MUT)
axB.text(1,lmb-.055,'고전 %.2f'%lmb,ha='center',fontsize=8,color=MUT)
axB.plot([0,1],[np.mean(mk),np.mean(lm)],color='#c9ced3',lw=1.2,ls='--',zorder=1)
axB.set_xticks([0,1]); axB.set_xticklabels(['마커 기반\n(21명 · 588창)','지형지물 기반\n(28명 · 784창)'],fontsize=8.6)
axB.set_xlim(-.42,1.42); axB.set_ylabel('MAE (회/분)',fontsize=9,color=SEC)
axB.yaxis.grid(True,color='#eef1f3',lw=.8)
axB.text(0,1.10,'B.  자르기가 모델보다 크게 좌우합니다',transform=axB.transAxes,
         fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axB.text(0,1.015,'같은 방식 안에서 모델을 바꾼 폭(%.2f)보다 자르기를 바꾼 폭(%.2f)이 더 큽니다'
         %(max(lm)-min(lm),np.mean(mk)-np.mean(lm)),transform=axB.transAxes,fontsize=8.6,color=SEC,va='bottom')

# C. 조기종료 에폭 — ANN vs SNN
axC=fig.add_axes([.565,.085,.41,.33]); st(axC)
labs=[]; vals=[]; cols2=[]
for r in R:
    if r['mode']!='landmark': continue
    labs.append('ANN' if r['kind']=='ANN' else NAME[r['enc']]); vals.append(r['stop_epochs'])
    cols2.append('#7f9fb0' if r['kind']=='ANN' else S1)
for i,(v,c) in enumerate(zip(vals,cols2)):
    axC.scatter([i]*len(v),v,s=36,color=c,zorder=3)
    axC.plot([i,i],[min(v),max(v)],color=c,lw=1.4,alpha=.45,zorder=2)
    axC.scatter([i],[np.mean(v)],marker='_',s=180,color=INK,zorder=4,lw=1.6)
axC.set_xticks(range(len(labs))); axC.set_xticklabels(labs,fontsize=8.4,rotation=18)
axC.set_ylabel('조기종료 에폭',fontsize=9,color=SEC)
axC.yaxis.grid(True,color='#eef1f3',lw=.8)
axC.text(0,1.10,'C.  학습이 언제 멈췄는가',transform=axC.transAxes,
         fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axC.text(0,1.015,'검증셋이 좋아지기를 멈춘 에폭 (4폴드 각각) · 가로줄은 평균 · 성적 좋은 인코딩일수록 오래 학습됩니다',
         transform=axC.transAxes,fontsize=8.6,color=SEC,va='bottom')

fig.text(.075,.945,'최종 결과 — 지형지물 자르기 + SNN(delta)에서 1.18 회/분',fontsize=15,color=INK,fontweight='bold')
fig.text(.075,.902,'학습/검증/평가 3분할 · 피험자 분리 4-fold · 검증셋 기준 조기종료(인내 10, 상한 60에폭) · 평가셋은 마지막에 한 번만 확인',
         fontsize=9.3,color=SEC)
fig.savefig('fig_final.png',facecolor=SURF)
print('ok')
