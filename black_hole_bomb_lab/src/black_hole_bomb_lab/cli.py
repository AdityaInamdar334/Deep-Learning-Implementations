"""Run deterministic toy experiments and export figures and metadata."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .models import (GAMMA_PER_S, PLATEAU_M, FREQ_HZ,FREQ_DRIFT_HZ, A0_M,
                     linear_amplitude,saturated_amplitude,mode_amplitudes,
                     coupled_energy,bicoherence_experiment,circulation_quanta)


def run(out, seed=11):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    t=np.linspace(0,110,1101)
    lin=linear_amplitude(t); sat=saturated_amplitude(t)
    a,h,dc=mode_amplitudes(t)
    energy=coupled_energy(t)
    fig,ax=plt.subplots(figsize=(9.6,5.6))
    ax.plot(t,lin*1e3,label='Linear prediction, no limit',lw=2)
    ax.plot(t,sat*1e3,label='Toy nonlinear saturation (calibrated plateau)',lw=2)
    ax.axhline(PLATEAU_M*1e3,ls=':',color='0.3',label='Reported plateau 3.06 mm')
    ax.axvline(80,ls='--',color='0.5',label='Reported saturation onset ~80 s')
    ax.set_ylim(0,7)
    ax.set(xlabel='Time (s)',ylabel='Fundamental amplitude (mm)',title='Same measured early-time growth rate; two different endings')
    ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'01_linear_vs_saturated.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.6,5.6))
    ax.semilogy(t,1e3*a,label='Fundamental m=1, illustrative')
    ax.semilogy(t,1e3*h,label='Harmonic m=2 scales as A² (chosen coefficient)')
    ax.semilogy(t,1e3*dc,label='Static deformation m=0 scales as A² (chosen coefficient)')
    ax.set(xlabel='Time (s)',ylabel='Illustrative mode amplitude (mm, log scale)',title='Why a quadratic harmonic initially grows at twice the rate')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/'02_modes_growth.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.6,5.6))
    ax.plot(t,energy['primary']*1e6,label='Primary proxy energy')
    ax.plot(t,energy['secondary']*1e6,label='Secondary proxy energy')
    ax.axvline(80,ls='--',color='0.5')
    ax.set(xlabel='Time (s)',ylabel='Scaled envelope-energy proxy (mm²)',title='Hypothetical transfer-and-loss closure; not measured energy')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'03_energy_transfer_toy.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.6,5.6))
    gain=energy['injection']*1e6
    transfer=energy['transfer']*1e6
    loss=energy['dissipation']*1e6
    ax.plot(t,gain,label='Input to primary (illustrative)')
    ax.plot(t,transfer,label='Transfer out of primary (illustrative)')
    ax.plot(t,loss,label='Dissipation from secondary (illustrative)')
    ax.set(xlabel='Time (s)',ylabel='Rate of envelope-energy proxy (mm²/s)',title='Balancing energy rates in a two-mode toy model')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(out/'04_energy_rate_balance.png',dpi=180);plt.close(fig)

    coh=bicoherence_experiment(seed=seed)
    fig,ax=plt.subplots(figsize=(7,5))
    vals=[coh['coherent'],coh['random_phase']]
    ax.bar(['Locked harmonic','Independent phase'],vals,color=['#166b85','#b3b8bd'])
    ax.set_ylim(0,1.1);ax.set_ylabel('Normalized toy bicoherence')
    ax.set_title('Three-wave phase locking is testable, not just a visible harmonic')
    for i,v in enumerate(vals):ax.text(i,v+.02,f'{v:.3f}',ha='center')
    fig.tight_layout();fig.savefig(out/'05_bicoherence_control.png',dpi=180);plt.close(fig)

    # Time-dependent synthetic angular trace (not a reconstruction of the measured interface)
    fig,ax=plt.subplots(figsize=(9,5.5))
    th=np.linspace(0,2*np.pi,512,endpoint=False)
    for moment in [35,60,80,105]:
        A,B,Z=mode_amplitudes(moment)
        z=A*np.cos(th)+B*np.cos(2*th+.3)+Z
        ax.plot(np.degrees(th),z*1e3,label=f't={moment} s')
    ax.set(xlabel='Azimuthal angle (degrees)',ylabel='Synthetic surface displacement (mm)',title='A growing m=1 sloshing wave creates higher harmonics')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'06_synthetic_angular_wave.png',dpi=180);plt.close(fig)

    t2=np.array([0,25,45,65,80,95,110],float)
    values=saturated_amplitude(t2)
    result={'data_provenance':'Published values cited; all trajectories are mathematical toy output, not experimental measurements',
            'published_parameters':{'growth_rate_s_inverse':GAMMA_PER_S,'growth_rate_error_s_inverse':0.001,
            'plateau_mm':PLATEAU_M*1e3,'plateau_error_mm':0.05,'measured_frequency_hz':FREQ_HZ,
            'observed_frequency_drift_magnitude_hz':FREQ_DRIFT_HZ,'saturation_time_approx_s':80},
            'toy_parameters':{'initial_amplitude_um':A0_M*1e6,'model':'Stuart-Landau envelope, logistic energy transfer, non-paper coefficients','secondary_decay_s_inverse':0.65},
            'estimated_circulation_quanta':round(circulation_quanta()),
            'mode_1_toy_amplitudes_mm':{str(int(x)):float(y*1e3) for x,y in zip(t2,values)},
            'toy_bicoherence':coh}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savetxt(out/'toy_mode_timeseries.csv',np.column_stack([t,sat*1e3,lin*1e3,h*1e3,energy['primary']*1e6,energy['secondary']*1e6]),
               delimiter=',',header='time_s,toy_saturated_primary_mm,toy_linear_primary_mm,toy_second_harmonic_mm,toy_primary_energy_mm2,toy_secondary_energy_mm2',comments='')
    print(json.dumps(result,indent=2))
    print('Wrote six figures, CSV, and summary.json to',out)


def main():
    parser=argparse.ArgumentParser(description='Educational black-hole-bomb analogue models; no paper data or experimental replication')
    parser.add_argument('--out',default='results')
    parser.add_argument('--seed',type=int,default=11)
    args=parser.parse_args()
    run(args.out,args.seed)
if __name__=='__main__':main()
