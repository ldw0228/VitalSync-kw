# -*- coding: utf-8 -*-
"""시계열용 스파이크 인코딩 5종"""
import torch, math

class Direct:
    name='direct'; desc='아날로그 전류 주입 (기준선, 엄밀히는 스파이크 아님)'
    def dim(self,F): return F
    def __call__(self,x): return x

class Rate:
    name='rate'; desc='Bernoulli 레이트 부호화 (양/음 2채널 정류)'
    def __init__(self,hi=2.0): self.hi=hi
    def dim(self,F): return 2*F
    def __call__(self,x):
        pp=(x/self.hi).clamp(0,1); pn=(-x/self.hi).clamp(0,1)
        return torch.bernoulli(torch.cat([pp,pn],-1))

class Delta:
    name='delta'; desc='시간 대비(ON/OFF) — DVS 이벤트 방식'
    def __init__(self,th=0.3): self.th=th
    def dim(self,F): return 2*F
    def __call__(self,x):
        d=torch.zeros_like(x); d[:,1:]=x[:,1:]-x[:,:-1]
        return torch.cat([(d>self.th).float(),(d<-self.th).float()],-1)

class StepForward:
    name='sf'; desc='Step-Forward — 기준선을 추종하며 레벨을 부호화'
    def __init__(self,th=0.4): self.th=th
    def dim(self,F): return 2*F
    def __call__(self,x):
        B,T,F=x.shape; base=x[:,0].clone()
        on=torch.zeros(B,T,F,device=x.device); off=torch.zeros_like(on)
        for t in range(T):
            d=x[:,t]-base
            p=(d>self.th).float(); n=(d<-self.th).float()
            on[:,t]=p; off[:,t]=n
            base=base+self.th*p-self.th*n
        return torch.cat([on,off],-1)

class Population:
    name='pop'; desc='가우시안 수용장 모집단 부호화 (연속량 표준)'
    def __init__(self,M=5,lo=-2.0,hi=2.0,th=0.55):
        self.M=M; self.th=th
        self.mu=torch.linspace(lo,hi,M); self.sg=(hi-lo)/(M-1)*0.8
    def dim(self,F): return self.M*F
    def __call__(self,x):
        mu=self.mu.to(x.device).view(1,1,1,-1)
        a=torch.exp(-((x.unsqueeze(-1)-mu)**2)/(2*self.sg**2))
        s=(a>self.th).float()
        return s.flatten(-2)

ENC=[Direct(),Rate(),Delta(),StepForward(),Population()]
