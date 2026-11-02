"""Replot the real-station comparison from public data, without refitting or imputation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
MODELS = [('XGB29','XGB2.9G','#C54D3C','D'),('AB06','AB06','#287AA0','o'),
          ('ABMIX','AB10 + AB06','#7964A4','P'),('FIVE','Five-GMPE combination','#3E8863','X'),
          ('KOTHA','Kotha 2020','#A77828','^')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_output_directory',type=Path)
    args = parser.parse_args()
    if args.new_output_directory.exists():
        parser.error('Choose a NEW output directory')
    d = pd.read_csv(ROOT/'results/tables/real_recordings_station_predictions.csv')
    d = d.sort_values(['Mw','station'],ascending=[False,True]).reset_index(drop=True)
    table = pd.read_csv(ROOT/'results/tables/real_recordings_available_and_common_subset_metrics.csv')
    metrics = table[table.scope.eq('all_available_10_article_recordings')].set_index('model_code')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig = plt.figure(figsize=(10.7,11.9),layout='constrained')
    grid = fig.add_gridspec(3,2,height_ratios=[6.4,.65,2.8],hspace=.05,wspace=.07)
    ax = fig.add_subplot(grid[0,:])
    ax.axvspan(1/3,3,color='#F2F5F8');ax.axvspan(.5,2,color='#E3EBF2')
    ax.axvline(1,color='#374754',lw=.9)
    for i,(code,label,color,marker) in enumerate(MODELS):
        valid = d[code].notna()
        ax.scatter(d.loc[valid,code]/d.loc[valid,'observed_pga_pctg'],np.flatnonzero(valid)+(i-2)*.095,
                   color=color,marker=marker,s=44,edgecolor='white',linewidth=.5,label=label,zorder=3)
    labels = [f'{r.display_case.split()[0]} '+rf'$M_w$ {r.Mw:.1f}'+'\n'+r.station for r in d.itertuples()]
    ax.set_yticks(np.arange(10),labels);ax.set_ylim(9.6,-.6)
    ax.set_xscale('log');ax.set_xlim(.025,4)
    ax.xaxis.set_major_locator(FixedLocator([.03,.1,.2,.5,1,2,3]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'));ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel('Prediction / observed station PGA');ax.set_title('(a) Station predictions',loc='left',fontsize=11)
    ax.grid(axis='x',color='#DFE6EC',lw=.6);ax.set_axisbelow(True)
    legend = fig.add_subplot(grid[1,:]);legend.axis('off')
    legend.legend(*ax.get_legend_handles_labels(),loc='center',ncols=3,frameon=False,fontsize=9.5)
    for j,(field,label) in enumerate([('MAE_pctg','MAE'),('RMSE_pctg','RMSE')]):
        bar = fig.add_subplot(grid[2,j]);values = metrics.loc[[m[0] for m in MODELS],field].to_numpy()
        bars = bar.barh(range(5),values,color=[m[2] for m in MODELS],height=.57)
        bar.invert_yaxis();bar.set_xlim(0,values.max()*1.2)
        bar.set_yticks(range(5),[f'{m[1]}  (n = {int(metrics.loc[m[0],"n"])})' for m in MODELS] if j==0 else ['']*5,fontsize=9.5)
        bar.set_xlabel(label+' (%g)');bar.set_title(f'({chr(98+j)}) {label}: all available cases',loc='left',fontsize=10.5)
        bar.grid(axis='x',color='#DFE6EC',lw=.6);bar.set_axisbelow(True);bar.bar_label(bars,fmt='%.2f',padding=5)
    args.new_output_directory.mkdir(parents=True)
    fig.savefig(args.new_output_directory/'real_recordings_all_available.png',dpi=600,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    main()
