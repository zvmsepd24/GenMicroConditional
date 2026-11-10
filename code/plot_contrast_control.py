"""Plot the stronger-response control from saved numerical results, without checkpoints."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent.parent


def plot(output=None):
    d=json.loads((ROOT/'results'/'contrast_control.json').read_text())
    out=Path(output) if output else ROOT/'figures'
    out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    u=np.linspace(-1,1,1201)
    for a,c,ls in [(.15,'#777777','--'),(.8,'#111111','-')]:
        ax[0].plot(u,a*(1-np.cos(8*np.pi*u))/2,color=c,ls=ls,label=f'$a={a:g}$')
    ax[0].set(xlabel='$u_1$',ylabel='$w_R$')
    ax[0].legend(frameon=False)
    ax[1].plot(d['line_u'],d['line_exact_g11'],color='black',label='Exact')
    for name,c,label in [('osc_strong_m8','#2a78d6','Sinusoidal'),('osc_strong_m8_raw','#eb6834','Direct')]:
        q=d['models'][name]
        ax[1].plot(np.array(q['u'])[:,0],np.array(q['G'])[:,0,0],'-o',ms=3,color=c,label=label)
    ax[1].set(xlabel='$u_1$',ylabel=r'$G_{11},\;\widehat G_{11}$',yscale='log')
    ax[1].legend(frameon=False)
    for a in ax:
        a.spines[['top','right']].set_visible(False)
    fig.savefig(out/'contrast_control.png',dpi=250)
    fig.savefig(out/'contrast_control.pdf')
    plt.close(fig)


if __name__=='__main__':plot()
