# -*- coding: utf-8 -*-
"""파싱 오류 vs 올바른 파싱, 그리고 진짜 멀티패스 — 회의용 그림"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager as fm
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d

for c in ['Noto Sans CJK KR','Noto Sans CJK JP','Noto Sans CJK SC']:
    if any(c==f.name for f in fm.fontManager.ttflist): plt.rcParams['font.family']=c; break
plt.rcParams['axes.unicode_minus']=False

SURF='#fcfcfb'; INK='#0b0b0b'; SEC='#52514e'; MUT='#8a8983'
S1='#2a78d6'; S2='#eb6834'; S3='#1baf7a'
SEQ=LinearSegmentedColormap.from_list('seq',[SURF,'#cfe0f5','#7fb0e8',S1,'#1a4b8c','#0d2643'])

z=np.load('_work/_raw184_S03.npz')
FS=10.0
def bp(x,lo=0.1,hi=0.6,fs=FS,o=3):
    b,a=butter(o,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x,axis=0)
def env(x):  # 호흡대역 진폭 포락선
    return uniform_filter1d(np.abs(bp(x)),int(6*FS),axis=0)

R='r3'
m=z[R].astype(float); m=m-m.mean(0,keepdims=True)
E_flat=env(m)                                   # (T,184) 잘못된 보기
c=m[:,:92]+1j*m[:,92:]; c=c-c.mean(0,keepdims=True)
E_cplx=env(np.abs(c))                           # (T,92) 올바른 보기
t=np.arange(m.shape[0])/FS

fig=plt.figure(figsize=(11.6,7.4),dpi=200)
fig.patch.set_facecolor(SURF)
gs=fig.add_gridspec(2,2,height_ratios=[1.25,1.0],hspace=.50,wspace=.20,
                    left=.065,right=.985,top=.845,bottom=.085)

def hm(ax,E,ymax,title,sub):
    v=np.percentile(E,99.3)
    ax.imshow(E.T,aspect='auto',origin='lower',cmap=SEQ,vmin=0,vmax=v,
              extent=[t[0],t[-1],0,ymax],interpolation='nearest')
    ax.text(0,1.105,title,transform=ax.transAxes,fontsize=11.5,color=INK,
            va='bottom',fontweight='bold')
    ax.text(0,1.025,sub,transform=ax.transAxes,fontsize=8.8,color=SEC,va='bottom')
    ax.set_xlabel('시간 (초)',fontsize=9,color=SEC)
    ax.set_ylabel('거리 bin',fontsize=9,color=SEC)
    ax.tick_params(colors=SEC,labelsize=8.5,length=3)
    for s in ax.spines.values(): s.set_color('#dfe3e8')

# ---- A. 잘못된 보기
axA=fig.add_subplot(gs[0,0])
hm(axA,E_flat,184,'A.  184개 값을 그대로 거리축으로 볼 때',
   '선배가 보신 화면 — 위아래로 한 쌍이 뜹니다')
axA.axhline(92,color=S2,lw=1.4,ls='--')
axA.text(t[-1]*.985,94,'블록 경계 (92)',ha='right',va='bottom',fontsize=8.2,color=S2)
k1,k2=31,123
for k,lb in [(k1,'아래 줄'),(k2,'위 줄')]:
    axA.annotate('',xy=(t[-1]*.055,k),xytext=(t[-1]*-.02,k),
                 arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.3))
    axA.text(t[-1]*.075,k,'%s  bin %d'%(lb,k),fontsize=8.6,color=INK,va='center',fontweight='bold')
axA.annotate('',xy=(t[-1]*.90,k1),xytext=(t[-1]*.90,k2),
             arrowprops=dict(arrowstyle='<|-|>',color=S2,lw=1.5))
axA.text(t[-1]*.885,(k1+k2)/2,'간격 = 92\n(블록 크기와 정확히 같음)',ha='right',va='center',
         fontsize=8.6,color=S2,fontweight='bold')
# 2차 반사(62)도 똑같이 154로 복제된다
axA.annotate('',xy=(t[-1]*.335,62),xytext=(t[-1]*.335,154),
             arrowprops=dict(arrowstyle='<|-|>',color=MUT,lw=1.1))
axA.text(t[-1]*.320,108,'2차 반사도 같은 92 간격으로\n한 번 더 나타납니다  (62 ↔ 154)',
         ha='right',va='center',fontsize=8.2,color=SEC)

# ---- B. 올바른 보기
axB=fig.add_subplot(gs[0,1])
hm(axB,E_cplx,92,'B.  [실수 92][허수 92]로 나눠 복소수로 합칠 때',
   '한 줄로 모이고, 그 위에 진짜 2차 반사가 남습니다')
ec=E_cplx.mean(0); p1=int(np.argmax(ec))
mask=np.ones(92,bool); mask[max(p1-6,0):p1+7]=False
p2=int(np.arange(92)[mask][np.argmax(ec[mask])])
axB.annotate('',xy=(t[-1]*.055,p1),xytext=(t[-1]*-.02,p1),
             arrowprops=dict(arrowstyle='-|>',color=INK,lw=1.3))
axB.text(t[-1]*.075,p1,'가슴  bin %d'%p1,fontsize=8.6,color=INK,va='center',fontweight='bold')
axB.annotate('2차 반사  bin %d  (+%d)'%(p2,p2-p1),xy=(t[-1]*.42,p2),
             xytext=(t[-1]*.30,p2+21),fontsize=8.6,color=S2,fontweight='bold',
             arrowprops=dict(arrowstyle='-|>',color=S2,lw=1.3))

# ---- C. 거리 프로파일
axC=fig.add_subplot(gs[1,:])
axC.set_facecolor(SURF)
prof={}
for i,(rk,col) in enumerate(zip(['r1','r2','r3'],[S1,S2,S3])):
    mm=z[rk].astype(float); mm=mm-mm.mean(0,keepdims=True)
    cc=mm[:,:92]+1j*mm[:,92:]; cc=cc-cc.mean(0,keepdims=True)
    e=bp(np.abs(cc)).std(0); e=e/e.max()
    prof[rk]=e
    axC.plot(np.arange(92),e,color=col,lw=2.0,solid_capstyle='round')
    pk=int(np.argmax(e))
    dx,dy,ha=[(-6,.10,'right'),(7,.02,'left'),(2,.115,'left')][i]
    axC.text(pk+dx,e[pk]+dy,'레이더 %d'%(i+1),color=col,fontsize=9.5,ha=ha,fontweight='bold')
e3=prof['r3']; p1=int(np.argmax(e3))
mk=np.ones(92,bool); mk[max(p1-6,0):p1+7]=False
p2=int(np.arange(92)[mk][np.argmax(e3[mk])])
axC.plot([p2],[e3[p2]],'o',ms=9,mfc='none',mec=S3,mew=2.2)
axC.annotate('레이더 3의 2차 반사  bin %d (+%d)\n29명 전원 +28~+32로 일관 · 세기 0.43\n→ 진짜 멀티패스 후보'%(p2,p2-p1),
             xy=(p2,e3[p2]),xytext=(p2+7,e3[p2]+.30),fontsize=9,color=INK,
             arrowprops=dict(arrowstyle='-|>',color=SEC,lw=1.2))
axC.set_xlim(0,91); axC.set_ylim(0,1.15)
axC.set_xlabel('거리 bin  (올바른 파싱, 복소 92)',fontsize=9,color=SEC)
axC.set_ylabel('호흡대역 세기 (최대=1)',fontsize=9,color=SEC)
axC.text(0,1.075,'C.  올바른 파싱에서의 거리 프로파일 — 레이더 3대',transform=axC.transAxes,
         fontsize=11.5,color=INK,va='bottom',fontweight='bold')
axC.tick_params(colors=SEC,labelsize=8.5,length=3)
axC.yaxis.grid(True,color='#e8ebee',lw=.8); axC.set_axisbelow(True)
for s in ['top','right']: axC.spines[s].set_visible(False)
for s in ['left','bottom']: axC.spines[s].set_color('#dfe3e8')

fig.text(.065,.962,'IR-UWB 원시 데이터 파싱 — "고스트"의 정체',fontsize=15.5,color=INK,fontweight='bold')
fig.text(.065,.930,'S03_PSJ · 1번 실험 240초 · 호흡대역 0.1~0.6 Hz 필터 후 진폭',fontsize=9.5,color=SEC)
fig.savefig('fig_ghost.png',facecolor=SURF)
print('ok  p1=%d p2=%d'%(p1,p2))
