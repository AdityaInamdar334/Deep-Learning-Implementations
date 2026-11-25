"""Reproducible linear algebra experiments accompanying the article."""
import argparse, json
from pathlib import Path
import numpy as np

def transform_demo():
    matrix=np.array([[2.0,0.8],[0.8,1.0]])
    vals,vecs=np.linalg.eigh(matrix)
    return matrix,vals,vecs

def solve_demo():
    # Explicit deterministic perturbation along the matrix's weak singular direction
    A=np.diag([1.0,1e-5])
    x_true=np.ones(2)
    b=A@x_true
    b_pert=b+np.array([0.0,1e-6])
    x_new=np.linalg.solve(A,b_pert)
    return {"condition_number":float(np.linalg.cond(A)),"original_solution":x_true.tolist(),"perturbed_solution":x_new.tolist(),"relative_input_change":float(np.linalg.norm(b_pert-b)/np.linalg.norm(b)),"relative_solution_change":float(np.linalg.norm(x_new-x_true)/np.linalg.norm(x_true))}

def pca_demo(seed=7):
    rng=np.random.default_rng(seed)
    n=500
    latent=rng.normal(size=(n,2))*np.array([3.0,0.8])
    theta=np.deg2rad(32)
    rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    X=latent@rotation.T+np.array([2.0,-1.0])
    mean=X.mean(axis=0)
    centered=X-mean
    U,S,Vt=np.linalg.svd(centered,full_matrices=False)
    direction=Vt[0]
    score=centered@direction
    recon=np.outer(score,direction)+mean
    var_ratio=float(S[0]**2/np.sum(S**2))
    mse=float(np.mean((X-recon)**2))
    return X,recon,mean,direction,var_ratio,mse,S

def make_figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True,exist_ok=True)
    A,vals,V=transform_demo()
    t=np.linspace(0,2*np.pi,400)
    circle=np.stack([np.cos(t),np.sin(t)])
    mapped=A@circle
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    axes[0].plot(circle[0],circle[1],label='unit circle')
    axes[1].plot(mapped[0],mapped[1],label='transformed circle')
    for i in range(2):
        arrow=V[:,i]
        axes[0].arrow(0,0,*arrow,width=.015,head_width=.12,length_includes_head=True)
        axes[1].arrow(0,0,*(vals[i]*arrow),width=.025,head_width=.16,length_includes_head=True)
    for ax in axes:
        ax.set_aspect('equal');ax.axhline(0,linewidth=.5);ax.axvline(0,linewidth=.5);ax.grid(alpha=.3);ax.legend()
    axes[0].set_title('Before: unit circle and eigenvectors');axes[1].set_title('After: stretched and rotated')
    fig.tight_layout();fig.savefig(out/'matrix_geometry.png',dpi=160);plt.close(fig)
    X,R,mean,direction,var,mse,S=pca_demo()
    fig,ax=plt.subplots(figsize=(8,5.5))
    ax.scatter(X[:,0],X[:,1],s=11,alpha=.27,label='original points')
    ax.scatter(R[:,0],R[:,1],s=8,alpha=.35,label='rank-1 reconstruction')
    end=mean+3*direction
    ax.plot([mean[0]-3*direction[0],end[0]],[mean[1]-3*direction[1],end[1]],linewidth=2,label='first principal direction')
    ax.set_title(f'PCA: preserve {var:.1%} of centered variance with one direction')
    ax.set_xlabel('feature 1');ax.set_ylabel('feature 2');ax.legend();ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'pca_projection.png',dpi=160);plt.close(fig)
    ds=solve_demo()
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    vals=np.array([ds['relative_input_change'],ds['relative_solution_change']])
    ax.bar(['relative change in b','relative change in solution'],vals)
    ax.set_ylabel('relative L2 change');ax.set_title(f'Ill-conditioning amplifies a small perturbation (cond={ds["condition_number"]:,.0f})')
    ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(out/'conditioning.png',dpi=160);plt.close(fig)
    return {"eigenvalues":np.linalg.eigvalsh(A).tolist(),"pca_first_component_explained_variance":var,"pca_reconstruction_mse":mse,**ds}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='figures')
    args=parser.parse_args()
    out=Path(args.out)
    results=make_figures(out)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
