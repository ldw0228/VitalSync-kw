# -*- coding: utf-8 -*-
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for c in ['Noto Sans CJK KR','Noto Sans CJK JP','Noto Sans CJK SC','WenQuanYi Zen Hei']:
    if any(c==f.name for f in fm.fontManager.ttflist): plt.rcParams['font.family']=c; print('font:',c); break
plt.rcParams['axes.unicode_minus']=False
V=''
INK='#1f2933'; ACC='#2f6f8f'; ACC2='#c26b3f'; GRID='#dfe3e8'
def style(ax):
    for s in ['top','right']: ax.spines[s].set_visible(False)
    for s in ['left','bottom']: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9); ax.yaxis.grid(True,color=GRID,lw=.7); ax.set_axisbelow(True)

sa=json.load(open(V+'summary_A.json')); sb=json.load(open(V+'summary_B.json'))
# 1) 인코딩 비교
keys=['enc:rate','enc:sf','enc:pop','enc:delta','enc:direct']
lab=['rate','step-forward','population','delta','direct(아날로그)']
fig,ax=plt.subplots(figsize=(7.2,3.2),dpi=200)
x=np.arange(len(keys)); w=.36
a=[sa[k]['mean']*100 for k in keys]; ae=[sa[k]['std']*100 for k in keys]
b=[sb[k]['mean']*100 for k in keys]; be=[sb[k]['std']*100 for k in keys]
ax.bar(x-w/2,a,w,yerr=ae,capsize=3,color=ACC,label='과제 A · 자세 각도 (3클래스)')
ax.bar(x+w/2,b,w,yerr=be,capsize=3,color=ACC2,label='과제 B · 호흡 상태 (6클래스)')
ax.axhline(33.3,ls='--',lw=.9,color=ACC,alpha=.6); ax.axhline(16.7,ls='--',lw=.9,color=ACC2,alpha=.6)
ax.text(-0.45,34.6,'우연 33.3%',fontsize=7.5,color=ACC,ha='left')
ax.text(-0.45,18.0,'우연 16.7%',fontsize=7.5,color=ACC2,ha='left')
ax.set_xticks(x); ax.set_xticklabels(lab,fontsize=8.5); ax.set_ylabel('정확도 (%)',fontsize=9)
ax.set_ylim(0,80); ax.legend(fontsize=8,frameon=False,loc='upper right'); style(ax)
fig.tight_layout(); fig.savefig('c_enc.png'); plt.close(fig)

# 2) 각도별 호흡 변조 진폭
ph=json.load(open(V+'angle_phys.json'))
fig,ax=plt.subplots(figsize=(4.6,3.0),dpi=200)
ks=['0','45','90']; m=[ph[k]['amp'][0] for k in ks]; e=[1.96*ph[k]['amp'][1] for k in ks]
ax.bar(range(3),m,.55,yerr=e,capsize=4,color=[ACC,'#7f9fb0','#b9c6cd'])
for i,v in enumerate(m): ax.text(i,v+e[i]+.04,'%.2f'%v,ha='center',fontsize=9,color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(['0°(정면)','45°','90°(측면)'],fontsize=9)
ax.set_ylabel('정규화 호흡 변조 진폭',fontsize=9); ax.set_ylim(0,1.75); style(ax)
fig.tight_layout(); fig.savefig('c_phys.png'); plt.close(fig)

# 3) 과제 B 혼동행렬
cm=np.array(sb['enc:rate']['cm'],dtype=float); cmn=cm/cm.sum(1,keepdims=True)
nm=['평소\n호흡','느린\n호흡','무호흡','회복\n호흡','운동','운동 후\n호흡']
fig,ax=plt.subplots(figsize=(4.8,4.2),dpi=200)
im=ax.imshow(cmn,cmap='Blues',vmin=0,vmax=1)
for i in range(6):
    for j in range(6):
        ax.text(j,i,'%d'%round(cmn[i,j]*100),ha='center',va='center',fontsize=8.5,
                color='white' if cmn[i,j]>.5 else INK)
ax.set_xticks(range(6)); ax.set_xticklabels(nm,fontsize=7.5)
ax.set_yticks(range(6)); ax.set_yticklabels(nm,fontsize=7.5)
ax.set_xlabel('예측',fontsize=9); ax.set_ylabel('실제',fontsize=9)
ax.set_title('행 기준 %',fontsize=8.5,color='#6b7680')
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout(); fig.savefig('c_cmB.png'); plt.close(fig)

# 4) 회복 호흡 곡선
rc=json.load(open('recovery.json'))['stats']
lbl=list(rc); med=[rc[l]['amp_med'] for l in lbl]
fig,ax=plt.subplots(figsize=(6.2,2.9),dpi=200)
ax.plot(range(len(lbl)),med,'-o',color=ACC2,lw=2,ms=6)
ax.axhline(1.0,ls='--',lw=.9,color='#9aa5ad')
ax.axvspan(0.5,1.5,color=ACC2,alpha=.10)
for i,v in enumerate(med): ax.text(i,v+.035,'%.2f'%v,ha='center',fontsize=8.5,color=INK)
ax.set_xticks(range(len(lbl)))
ax.set_xticklabels(['평소 호흡\n(기준)','0~20초','20~40초','40~60초','60~80초'],fontsize=8.5)
ax.set_ylabel('호흡 진폭비 (중앙값)',fontsize=9); ax.set_ylim(.9,1.5); style(ax)
ax.text(1.0,1.43,'실제로 구분되는 구간',fontsize=8,color=ACC2,ha='center')
fig.tight_layout(); fig.savefig('c_recov.png'); plt.close(fig)
print('ok')
