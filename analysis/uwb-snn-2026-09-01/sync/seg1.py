# -*- coding: utf-8 -*-
import numpy as np
def find_block(mk, target, tol=25):
    """마커 목록에서 길이가 target 에 가장 가까운 (시작,끝) 쌍"""
    best=None
    for i in range(len(mk)):
        for j in range(i+1,len(mk)):
            d=mk[j]-mk[i]
            if abs(d-target)>tol: continue
            sc=abs(d-target)+0.02*mk[i]        # 앞쪽 우선
            if best is None or sc<best[0]: best=(sc,mk[i],mk[j])
    return (best[1],best[2]) if best else (None,None)
