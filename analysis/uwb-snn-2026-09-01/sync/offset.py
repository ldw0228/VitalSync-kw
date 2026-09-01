# -*- coding: utf-8 -*-
"""마커 검출 + offset 추정 (개선판)"""
import numpy as np, glob, os, json
from scipy.ndimage import median_filter, uniform_filter1d
W='_work'
FPS=40.0

def detect_markers(rsp, fsb, thr=8.5, merge=4.0):
    tb=np.arange(len(rsp))/fsb
    above=(rsp>thr).astype(int); dd=np.diff(np.r_[0,above,0])
    on=np.where(dd==1)[0]; off=np.where(dd==-1)[0]-1
    mk=[tb[s+int(np.argmax(rsp[s:e+1]))] for s,e in zip(on,off) if e>=s]
    if not mk: return np.array([])
    mk=sorted(mk); grp=[mk[0]]; out=[]
    for v in mk[1:]:
        if v-grp[-1]<=merge: grp.append(v)
        else: out.append(np.mean(grp)); grp=[v]
    out.append(np.mean(grp)); return np.array(out)

def contrast(mot, fps=FPS, smooth=0.5, base=15.0):
    """지역 배경 대비 움직임 돌출도"""
    s=uniform_filter1d(mot.astype(np.float64), int(smooth*fps))
    b=median_filter(s, size=int(base*fps), mode='nearest')
    c=s-b
    sd=c.std()+1e-9
    return c/sd

def estimate(mk, M, mode='contrast_sum', lo=-15, hi=15, step=0.05):
    """M: (nR, L) 움직임.  반환 (offset, 신뢰도, 곡선, cand)"""
    L=M.shape[1]; t=np.arange(L)/FPS
    if mode=='contrast_sum':
        feats=[contrast(m) for m in M]; F=np.sum(feats,axis=0)
    elif mode=='orig_mean':
        s=uniform_filter1d(M.mean(axis=0).astype(np.float64), int(0.5*FPS))
        F=(s-s.min())/(s.max()-s.min()+1e-9)
    cand=np.arange(lo,hi+1e-9,step)
    sc=np.array([np.interp(mk-c, t, F, left=0, right=0).mean() for c in cand])
    bi=int(np.argmax(sc)); off=float(cand[bi])
    # 신뢰도: 최고점과, 최고점에서 ±2초 밖의 차선 봉우리 차이 (표준편차 단위)
    mask=np.abs(cand-off)>2.0
    runner=sc[mask].max() if mask.any() else -np.inf
    conf=float((sc[bi]-runner)/(sc.std()+1e-9))
    return off, conf, sc, cand

def load(subj):
    z=np.load(os.path.join(W,subj+'_mot.npz'))
    if z['mot'].shape[0]==0 if 'mot' in z else True: return None
    return z

SUBS=sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(W,'*_mot.npz')))
if __name__=='__main__':
    print('subj        n마커  원본방식   개선방식  신뢰도  전반부   후반부   드리프트')
    rows=[]
    for subj in SUBS:
        z=np.load(os.path.join(W,subj+'_mot.npz'))
        if 'mot' not in z or z['mot'].shape[0]==0: print(subj,'레이더 없음'); continue
        M=z['mot']; rsp=z['rsp']; fsb=float(z['fsb'])
        mk=detect_markers(rsp,fsb)
        front=mk[mk<300]; front=front if len(front) else mk
        o_orig,_,_,_=estimate(front,M,'orig_mean',-12,12,0.1)
        o_new,conf,_,_=estimate(mk,M,'contrast_sum')
        half=len(mk)//2
        o_a,_,_,_=estimate(mk[:half],M,'contrast_sum') if half>2 else (np.nan,0,0,0)
        o_b,_,_,_=estimate(mk[half:],M,'contrast_sum') if len(mk)-half>2 else (np.nan,0,0,0)
        drift=o_b-o_a
        rows.append(dict(subj=subj,n=int(len(mk)),orig=o_orig,new=o_new,conf=conf,
                         a=float(o_a),b=float(o_b),drift=float(drift),
                         markers=[round(float(v),2) for v in mk]))
        print('%-10s %4d  %+8.2f  %+8.2f  %6.1f  %+7.2f %+7.2f  %+7.2f'%(
            subj,len(mk),o_orig,o_new,conf,o_a,o_b,drift))
    json.dump(rows, open('offsets.json','w'), ensure_ascii=False)
