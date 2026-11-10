"""Stronger oscillatory-response control, with matched direct/sinusoidal conditioning.

Run: python code/contrast_control.py train
     python code/contrast_control.py measure
Original measurements are never overwritten. Models use one matched pool and the original
nested data sizes/step schedule. New results describe this control, not the original amplitude.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import grf
import systems as S
import study as St
import fisher_v5 as F
from measure_v4 import load_model

ROOT=Path(__file__).resolve().parent.parent
RES=ROOT/'results'


def exact_g11(u, amplitude, frequency=8):
    comp=grf.mix_components(64,u.device)
    s=.25
    wr=amplitude*(1-torch.cos(torch.pi*frequency*u[:,0]))/2
    dwr=amplitude*torch.pi*frequency*torch.sin(torch.pi*frequency*u[:,0])/2
    base=torch.stack([1+s*u[:,0],1-s*u[:,0],1+s*u[:,1],1-s*u[:,1]],1)/4
    db=torch.tensor([s,-s,0.,0.],device=u.device,dtype=u.dtype)[None]/4
    weights=torch.cat([base*(1-wr[:,None]),wr[:,None]],1)
    dw=torch.cat([db*(1-wr[:,None])-base*dwr[:,None],dwr[:,None]],1)
    lam=torch.einsum('bj,jxy->bxy',weights,comp)
    dl=torch.einsum('bj,jxy->bxy',dw,comp)
    return .5*(dl/lam).square().sum((1,2))


def train():
    torch.backends.cudnn.benchmark=True
    for name in ['osc_strong_m8','osc_strong_m8_raw']:
        target=RES/f'study_{name}_0.json'
        if target.exists() and (RES/f'model_{name}_0.pt').exists():
            print(name,'already complete',flush=True)
            continue
        print('START',name,flush=True)
        torch.manual_seed(3400)
        St.run(name,0)


def measure():
    # Resolve the doubled frequency of squared response derivatives along the line.
    phases=torch.linspace(-.95,.95,65,dtype=torch.float64)
    u=torch.stack([phases,torch.zeros_like(phases)],1)
    exact={str(a):exact_g11(u.cuda(),a).cpu().tolist() for a in [.15,.8]}
    # Use all 65 line settings for both encodings and compare to the exact metric there.
    line=torch.stack([torch.linspace(-.95,.95,601,dtype=torch.float64),torch.zeros(601,dtype=torch.float64)],1)
    out=dict(amplitude=.8,original_amplitude=.15,frequency=8,s=.25,split=0,initialization_seed=3400,
             u=u.tolist(),exact_g11=exact,line_u=line[:,0].tolist(),
             line_exact_g11=exact_g11(line.cuda(),.8).cpu().tolist(),models={})
    # Fixed sample/noise seeds across encodings; 32 real images and the original sigma grid.
    pick=np.arange(len(u))
    for name in ['osc_strong_m8','osc_strong_m8_raw']:
        model=load_model(name,0)
        g,integ,_=F.grid_model_fisher(model,S.SYSTEMS[name],u[pick],R=32,seed=18300)
        ratios=g[:,0,0]/torch.tensor(exact['0.8'])[pick]
        out['models'][name]=dict(u=u[pick].tolist(),G=g.tolist(),
            sum_g11_ratio=float(g[:,0,0].sum()/torch.tensor(exact['0.8'])[pick].sum()),
            relative_g11_line_error=float(torch.linalg.vector_norm(g[:,0,0]-torch.tensor(exact['0.8'])[pick])/
                                          torch.linalg.vector_norm(torch.tensor(exact['0.8'])[pick])),
            per_setting_g11_ratio=ratios.tolist())
        print(name,'directional ratio',out['models'][name]['sum_g11_ratio'],flush=True)
        del model
    # Visibility is measured, not selected by choosing a favorable realization.
    locations=torch.tensor([[0.,0.],[.125,0.]],dtype=torch.float64)
    vis={}
    for a in [.15,.8]:
        images=[]
        for row in locations:
            images.append(grf.mix_sample(row[None].repeat(512,1),.25,a,8,
                generator=torch.Generator(device='cuda').manual_seed(52)).cpu())
        lo,hi=images
        rms=((lo-hi).square().mean((1,2))).sqrt()
        vis[str(a)]=dict(paired_rms_mean=float(rms.mean()),paired_rms_sd=float(rms.std()),
                        samples=512)
    out['visibility']=vis
    (RES/'contrast_control.json').write_text(json.dumps(out,indent=1))
    print('visibility',vis,flush=True)


if __name__=='__main__':
    RES.mkdir(exist_ok=True)
    {'train':train,'measure':measure}[sys.argv[1]]()
