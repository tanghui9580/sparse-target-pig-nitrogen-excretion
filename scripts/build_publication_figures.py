#!/usr/bin/env python3
"""Regenerate manuscript figures from the packaged machine-readable figure sources.

The generated figures are data-faithful reproducibility outputs. The locked submission artwork
is retained separately under publication_reference/artwork because exact typography/layout is
part of the publication layer rather than the statistical analysis layer.
"""
from __future__ import annotations
from pathlib import Path
import argparse, math, textwrap, os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'figure_sources'

COLORS = {
    'blue':'#2C7FB8','orange':'#E68613','green':'#4C9F70','red':'#D95F59','purple':'#7A58B5',
    'gray':'#8C9398','dark':'#30343B','lightgray':'#D9DEE2','teal':'#3B8D82','brown':'#A56A43',
    'pink':'#C76BAA','black':'#222222'
}
MODEL_COLORS={'MLR':COLORS['blue'],'RF':COLORS['orange'],'TabM':COLORS['green'],'MLR+RF':COLORS['red']}
MODEL_MARKERS={'MLR':'o','RF':'s','TabM':'^','MLR+RF':'D'}
INGREDIENT_COLORS={
    'Barley':'#80B1D3','Brown rice':'#B3DE69','Cassava':'#FDB462','Paddy rice':'#BEBADA',
    'Rapeseed cake':'#FB8072','Rice bran':'#8DD3C7','Sorghum':'#FCCDE5','Wheat':'#BC80BD','Wheat bran':'#CCEBC5'
}
METHOD_STYLE={
    'CG-HPP':(COLORS['purple'],'o','-'), 'Mean':(COLORS['brown'],'o','-'), 'Affine':(COLORS['orange'],'o','-'),
    'NoUpdate':(COLORS['gray'],'o','-'), 'TargetOnlyRidge':(COLORS['green'],'o','--'),
    'TargetOnlyElasticNet':(COLORS['pink'],'o','--'), 'TargetOnlyRF':(COLORS['red'],'o','--'),
    'TargetWeightedFullRF':(COLORS['teal'],'o','-')
}
METHOD_NAME_MAP={
    'CG-HPP update':'CG-HPP', 'Mean residual correction':'Mean',
    'Independent affine residual correction':'Affine',
    'Historical model (no target-domain update)':'NoUpdate',
    'Target-only Ridge':'TargetOnlyRidge','Target-only Elastic Net':'TargetOnlyElasticNet',
    'Target-only Random Forest':'TargetOnlyRF','Target-weighted full Random Forest revision':'TargetWeightedFullRF'
}

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':8.5,'axes.titlesize':10,'axes.labelsize':8.5,
    'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.2,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,
    'figure.dpi':120
})

def panel_label(ax, label, x=-0.13, y=1.05):
    ax.text(x,y,label,transform=ax.transAxes,fontweight='bold',fontsize=11,va='top')

def style_ax(ax, grid=True):
    if grid: ax.grid(True, axis='y', color='#E7E9EB', linewidth=.7, zorder=0)
    ax.set_axisbelow(True)

def save(fig, outdir:Path, stem:str, dpi=300):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir/f'{stem}.png', dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(outdir/f'{stem}.eps', bbox_inches='tight', facecolor='white')
    plt.close(fig)

def read(name, **kwargs): return pd.read_csv(SRC/name, **kwargs)

def figure1(out):
    nodes=read('Figure1_nodes.csv').set_index('node_id')
    fig,ax=plt.subplots(figsize=(8.2,10.0)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    bands=[('1  Predictor-information evaluation and model development',.73,.97,'#EFF5F9'),
           ('2  Direct evaluation in the target domain',.54,.70,'#FFF2EB'),
           ('3  Model updating with limited target measurements',.27,.51,'#EEF8F3'),
           ('4  Robustness analyses',.08,.24,'#F3F1FA')]
    for title,y0,y1,c in bands:
        ax.add_patch(FancyBboxPatch((.04,y0),.92,y1-y0,boxstyle='round,pad=.006',fc=c,ec='#98A2AA',lw=.8))
        ax.text(.055,y1-.018,title,fontsize=9.5,fontweight='bold',va='top')
    def box(node,x,y,w=.19,h=.052,fc='white',ec='#87939B',fs=6.6):
        r=nodes.loc[node]; ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=.006',fc=fc,ec=ec,lw=.8,zorder=3))
        label='\n'.join(textwrap.wrap(str(r.display_label),24)); ax.text(x,y,label,ha='center',va='center',fontsize=fs,fontweight='bold',zorder=4)
    def arrow(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=.8,color='#6F777D',zorder=2))
    # Stage 1: two clean rows.
    coords={
        'development673':(.15,.885),'predictor_layers':(.37,.885),'p2specified':(.59,.885),'benchmark':(.81,.885),
        'winner':(.22,.785),'confirmation233':(.50,.785),'all906':(.78,.785),
        'zero546':(.22,.615),'failure':(.50,.615),'shift':(.78,.615),
        'sampling':(.22,.415),'updating':(.50,.415),'budget':(.78,.415),'selectedproc':(.50,.315),
        'arch546':(.22,.155),'sow80':(.50,.155),'measurementdepth':(.78,.155)}
    for n,(x,y) in coords.items():
        fc='white'; ec='#87939B'; w=.20; h=.052
        if n in ['p2specified','winner','selectedproc']: fc='#DDEBF3'; ec='#5E8FAB'
        if n=='failure': fc='#FBE6DE'; ec='#C97655'
        if n in ['sampling','updating','budget']: fc='#F9FCFA'; ec='#77A48C'; h=.062
        if n=='selectedproc': fc='#DCEFE5'; ec='#5D9B79'; w=.25
        if n in ['arch546','sow80','measurementdepth']: w=.22
        box(n,x,y,w,h,fc,ec,6.3)
    # Primary path and branches.
    for a,b in [('development673','predictor_layers'),('predictor_layers','p2specified'),('p2specified','benchmark')]:
        arrow((coords[a][0]+.10,coords[a][1]),(coords[b][0]-.10,coords[b][1]))
    arrow((coords['benchmark'][0],coords['benchmark'][1]-.03),(coords['winner'][0]+.08,coords['winner'][1]+.03))
    arrow((coords['winner'][0]+.10,coords['winner'][1]),(coords['confirmation233'][0]-.10,coords['confirmation233'][1]))
    arrow((coords['confirmation233'][0]+.10,coords['confirmation233'][1]),(coords['all906'][0]-.10,coords['all906'][1]))
    arrow((coords['all906'][0],coords['all906'][1]-.03),(coords['zero546'][0]+.06,coords['zero546'][1]+.03))
    arrow((coords['zero546'][0]+.10,coords['zero546'][1]),(coords['failure'][0]-.10,coords['failure'][1]))
    arrow((coords['failure'][0]+.10,coords['failure'][1]),(coords['shift'][0]-.10,coords['shift'][1]))
    for n in ['sampling','updating','budget']:
        arrow((coords['failure'][0],coords['failure'][1]-.03),(coords[n][0],coords[n][1]+.035))
        arrow((coords[n][0],coords[n][1]-.035),(coords['selectedproc'][0],coords['selectedproc'][1]+.035))
    for n in ['arch546','sow80','measurementdepth']:
        arrow((coords['selectedproc'][0],coords['selectedproc'][1]-.035),(coords[n][0],coords[n][1]+.035))
    ax.text(.50,.035,'Outcome definitions: FN and UN modeled directly; RN = N intake − FN − UN; TNE = FN + UN',ha='center',fontsize=6.8,color='#666')
    save(fig,out,'Figure_1')

def figure2(out):
    a=read('Figure2A_Benchmark.csv'); b=read('Figure2B_Transport.csv'); c=read('Figure2C_UN.csv')
    models=(a[['rank','model','primary_nrmse']].drop_duplicates().sort_values('rank'))
    fig=plt.figure(figsize=(10.6,5.0)); gs=fig.add_gridspec(1,5,width_ratios=[1.45,.9,.9,1.35,1.15],wspace=.55)
    ax0=fig.add_subplot(gs[0]); ax1=fig.add_subplot(gs[1]); ax2=fig.add_subplot(gs[2]); ax3=fig.add_subplot(gs[3]); ax4=fig.add_subplot(gs[4])
    y=np.arange(len(models))[::-1]
    def mcolor(m):
        if m=='MLR+RF': return COLORS['red']
        if m in ['MLR','RF','TabM']: return MODEL_COLORS[m]
        return '#AEB5B9'
    ax0.scatter(models.primary_nrmse,y,c=[mcolor(x) for x in models.model],s=34,zorder=3)
    ax0.set_yticks(y,models.model); ax0.set_xlabel('Mean NRMSE\n(lower is better)'); ax0.set_title('Model comparison',fontweight='bold'); style_ax(ax0)
    panel_label(ax0,'A')
    for ax,endpoint in [(ax1,'FN'),(ax2,'UN')]:
        d=a[(a.endpoint==endpoint)&(a.metric=='R2')].merge(a[(a.endpoint==endpoint)&(a.metric=='RMSE_g_d')][['model','value']],on='model',suffixes=('_r2','_rmse'))
        d=d.sort_values('rank'); yy=np.arange(len(d))[::-1]
        ax.scatter(d.value_r2,yy,c=[mcolor(x) for x in d.model],s=30,zorder=3)
        for x,yv,rm in zip(d.value_r2,yy,d.value_rmse): ax.text(x+.004,yv,f'{rm:.2f}',fontsize=6.2,va='center',color='#777')
        ax.set_yticks(yy,[]); ax.set_xlabel(f'{endpoint} R²\nnumber = RMSE, g/d'); ax.set_title(f'{endpoint} performance',fontweight='bold'); style_ax(ax)
    for m in ['MLR','RF','TabM','MLR+RF']:
        row=b[b.model==m].iloc[0]; ax3.plot([0,1],[row.confirmation233_primary_nrmse,row.local546_primary_nrmse],marker=MODEL_MARKERS[m],color=MODEL_COLORS[m],lw=1.5,label=m)
    ax3.set_xticks([0,1],['Held-out\n233','546 target\nbefore updating']); ax3.set_ylabel('Mean NRMSE'); ax3.set_title('Evaluation performance',fontweight='bold',fontsize=9.2); style_ax(ax3); panel_label(ax3,'B',x=-.20,y=1.07)
    for m in ['MLR','RF','TabM','MLR+RF']:
        row=c[c.model==m].iloc[0]; yi=['MLR','MLR+RF','TabM','RF'].index(m); ax4.hlines(yi,row.R2,0,color=MODEL_COLORS[m],lw=1.4); ax4.plot(row.R2,yi,MODEL_MARKERS[m],color=MODEL_COLORS[m],ms=5)
    ax4.axvline(0,color='#888',lw=.7); ax4.set_yticks(range(4),['MLR','MLR+RF','TabM','RF']); ax4.set_xlabel('UN R² in 546 target\nbefore updating'); ax4.set_title('UN performance\nbefore updating',fontweight='bold',fontsize=9.2); panel_label(ax4,'C',x=-.22,y=1.07)
    fig.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=COLORS['red'],label='Selected: MLR+RF'),
                        Line2D([0],[0],marker='o',color='none',markerfacecolor=COLORS['blue'],label='Comparators: MLR/RF/TabM'),
                        Line2D([0],[0],marker='o',color='none',markerfacecolor='#AEB5B9',label='Other candidates')],loc='lower center',ncol=3,frameon=False)
    fig.subplots_adjust(bottom=.19)
    save(fig,out,'Figure_2')

def figure3(out):
    d=read('Figure3_budget_recovery_3_21.csv'); d=d[d.Architecture.eq('MLR+RF') & d.Method.isin(['NoUpdate','CG-HPP'])]
    fig,axs=plt.subplots(2,2,figsize=(8.4,7.2),sharex=True)
    specs=[('FN_R2_mean','Fecal nitrogen (FN)','R²'),('FN_RMSE_mean','Fecal nitrogen (FN)','RMSE (g/d)'),('UN_R2_mean','Urinary nitrogen (UN)','R²'),('UN_RMSE_mean','Urinary nitrogen (UN)','RMSE (g/d)')]
    for i,(ax,(col,title,yl)) in enumerate(zip(axs.flat,specs)):
        ax.axvspan(12.01,21.7,color='#F0F1F2',zorder=0); ax.axvline(12,color='#B8BDC0',lw=.7,ls=':')
        if i==1: ax.text(.98,1.06,'Post hoc extension',transform=ax.transAxes,ha='right',color='#777')
        for meth,color,label in [('NoUpdate',COLORS['blue'],'No target-data update'),('CG-HPP',COLORS['orange'],'CG-HPP updating')]:
            g=d[d.Method==meth].sort_values('Budget'); pre=g[g.Budget<=12]; post=g[g.Budget>=12]
            ax.plot(pre.Budget,pre[col],'-o',color=color,lw=1.6,ms=4,label=label)
            ax.plot(post.Budget,post[col],'--o',color=color,lw=1.4,ms=4,mfc='white')
        ax.set_title(title,fontweight='bold'); ax.set_ylabel(yl); ax.set_xticks([3,6,9,12,15,18,21]); style_ax(ax); panel_label(ax,chr(65+i))
        if 'R2' in col: ax.axhline(0,color='#999',ls=':',lw=.8)
    axs[1,0].set_xlabel('Measured source-specific diets'); axs[1,1].set_xlabel('Measured source-specific diets'); axs[0,0].legend(frameon=False,loc='lower right')
    save(fig,out,'Figure_3')

def plot_selection_panel(ax,d,budget):
    g=d[d.measured_source_specific_diets.eq(budget)]
    base=g[g.sampling_strategy.eq('Response-surface-guided sampling')].drop_duplicates('source_specific_diet_id')
    for ing,gg in base.groupby('ingredient_class'):
        ax.scatter(gg.PC1,gg.PC2,s=15,color=INGREDIENT_COLORS.get(ing,'#BBB'),alpha=.45,edgecolors='none')
    pam=g[g.sampling_strategy.str.startswith('P2-PAM')]; rs=g[g.sampling_strategy.eq('Response-surface-guided sampling')]; rnd=g[g.sampling_strategy.eq('Random sampling')]
    ax.scatter(pam.loc[pam.selected_or_selection_frequency>0,'PC1'],pam.loc[pam.selected_or_selection_frequency>0,'PC2'],marker='^',s=30,color=COLORS['blue'],edgecolor='white',lw=.3,label='P2-PAM')
    ax.scatter(rs.loc[rs.selected_or_selection_frequency>0,'PC1'],rs.loc[rs.selected_or_selection_frequency>0,'PC2'],marker='s',s=25,color=COLORS['purple'],edgecolor='white',lw=.3,label='RSGS')
    sizes=10+150*rnd.selected_or_selection_frequency.clip(0,.2)
    ax.scatter(rnd.PC1,rnd.PC2,marker='D',s=sizes,facecolors='none',edgecolors=COLORS['orange'],lw=.6,alpha=.75,label='Random')
    ax.set_title(f'{budget} diets',fontweight='bold'); ax.set_xlabel('PC1 (35.3%)'); style_ax(ax,False)

def figure4(out):
    sel=read('Figure4A_selection_all_budgets.csv'); sb=read('Figure4B_sampling_curves.csv'); uc=read('Figure4C_updating_curves.csv')
    fig=plt.figure(figsize=(10.7,7.4)); gs=fig.add_gridspec(2,4,height_ratios=[1.4,1],hspace=.65,wspace=.42)
    axes=[]
    for j,b in enumerate([3,6,9,12]):
        ax=fig.add_subplot(gs[0,j]); plot_selection_panel(ax,sel,b); axes.append(ax)
        if j==0: ax.set_ylabel('PC2 (24.1%)')
        else: ax.set_yticklabels([])
    panel_label(axes[0],'A'); axes[0].text(0,1.22,'Selected source-specific diets in target predictor space',transform=axes[0].transAxes,fontweight='bold',fontsize=10)
    axb=fig.add_subplot(gs[1,:2]); panel_label(axb,'B'); axb.set_title('Measurement selection — CG-HPP fixed',fontweight='bold')
    sstyles={'Response-surface-guided sampling':(COLORS['purple'],'o','RSGS'), 'P2-PAM representative sampling (k-medoids)':(COLORS['blue'],'s','P2-PAM'), 'Random sampling':(COLORS['gray'],'o','Random')}
    for name,(c,m,l) in sstyles.items():
        g=sb[sb['Measurement-selection strategy']==name].sort_values('Budget'); axb.plot(g.Budget,g['Mean-NRMSE'],marker=m,color=c,label=l,lw=1.5)
    axb.set_xlabel('Measured source-specific diets'); axb.set_ylabel('Mean NRMSE'); axb.set_xticks([3,6,9,12]); style_ax(axb); axb.legend(frameon=False)
    axc=fig.add_subplot(gs[1,2:]); panel_label(axc,'C'); axc.set_title('Model updating — response-surface-guided sampling fixed',fontweight='bold')
    keep=['CG-HPP','Mean','Affine','NoUpdate','TargetOnlyRidge','TargetWeightedFullRF']
    tmp=uc.copy(); tmp['Method']=tmp['Updating method / boundary'].map(METHOD_NAME_MAP)
    for m in keep:
        g=tmp[tmp.Method==m].sort_values('Budget'); color,marker,ls=METHOD_STYLE[m]; axc.plot(g.Budget,g['Mean-NRMSE'],marker=marker,color=color,ls=ls,lw=1.35,label=m.replace('TargetWeightedFullRF','Target-weighted RF').replace('NoUpdate','No update').replace('TargetOnlyRidge','Target-only Ridge'))
    axc.set_xlabel('Measured source-specific diets'); axc.set_ylabel('Mean NRMSE'); axc.set_xticks([3,6,9,12]); style_ax(axc); axc.legend(frameon=False,ncol=2)
    save(fig,out,'Figure_4')

def interval_h(ax,df,value,lo,hi,title,panel):
    order=['MLR','RF','TabM','MLR+RF']; y=np.arange(4)[::-1]
    for yi,m in zip(y,order):
        r=df[df.architecture==m].iloc[0]; ax.hlines(yi,r[lo],r[hi],color=MODEL_COLORS[m],lw=1.3); ax.plot(r[value],yi,MODEL_MARKERS[m],color=MODEL_COLORS[m],ms=5)
    ax.set_yticks(y,order); ax.set_xlabel('AUBC of mean NRMSE'); ax.set_title(title,fontweight='bold'); style_ax(ax); panel_label(ax,panel)

def figure5(out):
    a=read('Figure5A_546_AUBC.csv'); b=read('Figure5B_SOW80_AUBC.csv'); c=read('Figure5C_EndpointR2.csv'); d=read('Figure5D_DerivedEndpointR2.csv')
    fig=plt.figure(figsize=(10.5,8.2)); gs=fig.add_gridspec(3,4,height_ratios=[.9,1.15,1.05],hspace=.75,wspace=.55)
    axa=fig.add_subplot(gs[0,:2]); interval_h(axa,a,'AUBC_mean','allocation_q025','allocation_q975','546 target system — model-type robustness','A')
    axb=fig.add_subplot(gs[0,2:]); interval_h(axb,b,'AUBC_point','Study_cluster_CI95_lower','Study_cluster_CI95_upper','Gestating sows — 95% whole-study bootstrap interval','B')
    titles=[('546','FN','546 — FN'),('546','UN','546 — UN'),('SOW80','FN','Sows — FN'),('SOW80','UN','Sows — UN')]
    for j,(dom,ep,title) in enumerate(titles):
        ax=fig.add_subplot(gs[1,j]); panel_label(ax,chr(67+j)); ax.set_title(title,fontweight='bold')
        for m in ['MLR','RF','TabM','MLR+RF']:
            g=c[(c.domain==dom)&(c.endpoint==ep)&(c.architecture==m)].sort_values('measurement_budget'); ax.plot(g.measurement_budget,g.value,marker=MODEL_MARKERS[m],color=MODEL_COLORS[m],lw=1.2,ms=4,label=m)
        ax.axhline(0,color='#999',ls=':',lw=.7); ax.set_xticks([3,6,9,12]); ax.set_xlabel('Measured diets'); ax.set_ylabel('R²' if j in [0,2] else ''); style_ax(ax)
    for j,ep in enumerate(['RN','TNE']):
        ax=fig.add_subplot(gs[2,j*2:(j+1)*2]); panel_label(ax,chr(71+j)); ax.set_title(f'{ep} — mass-balance derived',fontweight='bold')
        for dom,color,ls,label in [('546',COLORS['black'],'-','546 target'),('SOW80',COLORS['gray'],'--','Gestating sow')]:
            g=d[(d.domain==dom)&(d.endpoint==ep)&(d.architecture=='MLR+RF')].sort_values('measurement_budget'); ax.plot(g.measurement_budget,g.value,marker='o',color=color,ls=ls,lw=1.3,ms=4,label=label)
        ax.set_xticks([3,6,9,12]); ax.set_xlabel('Measured diets'); ax.set_ylabel('R²'); style_ax(ax); ax.legend(frameon=False)
    fig.legend(handles=[Line2D([0],[0],color=MODEL_COLORS[m],marker=MODEL_MARKERS[m],label=m) for m in ['MLR','RF','TabM','MLR+RF']],loc='lower center',ncol=4,frameon=False)
    fig.subplots_adjust(bottom=.1)
    save(fig,out,'Figure_5')

def figureS1(out):
    d=read('FigureS01_DomainPCA.csv'); fig,axs=plt.subplots(1,2,figsize=(10,4.6),sharex=True,sharey=True)
    for ax,dom,accent,title in [(axs[0],'546 target',COLORS['purple'],'546 growing-pig target domain'),(axs[1],'SOW80',COLORS['green'],'80-record gestating-sow target domain')]:
        hist=d[d.Domain=='Historical 906']; tar=d[d.Domain==dom]
        ax.scatter(hist.PC1,hist.PC2,s=8,color='#D9DEE2',alpha=.55,edgecolors='none'); ax.scatter(tar.PC1,tar.PC2,s=18,color=accent,alpha=.8,edgecolors='none')
        ax.set_title(title,fontweight='bold'); ax.set_xlabel(f"PC1 ({d.PC1_explained_variance.iloc[0]*100:.1f}%)"); style_ax(ax,False)
    axs[0].set_ylabel(f"PC2 ({d.PC2_explained_variance.iloc[0]*100:.1f}%)"); panel_label(axs[0],'A'); panel_label(axs[1],'B'); save(fig,out,'Figure_S1')

def figureS2(out):
    d=read('FigureS02_DomainShift.csv'); vars=list(dict.fromkeys(d.Variable)); domains=['Held-out 233','546 target','SOW80']
    fig,axs=plt.subplots(1,3,figsize=(10.4,5.8),sharey=True)
    mats=[]
    for col in ['SMD_vs_Development673','Normalized_Wasserstein_vs_Development673','Pct_target_outside_Development_range']:
        mats.append(d.pivot(index='Variable',columns='Target_domain',values=col).reindex(index=vars,columns=domains))
    titles=['Standardized mean\ndifference','Normalized Wasserstein\ndistance','Outside development\nrange, %']; cmaps=['coolwarm','Blues','Reds']
    for i,(ax,p,title,cmap) in enumerate(zip(axs,mats,titles,cmaps)):
        arr=p.values
        if i==0: im=ax.imshow(arr,aspect='auto',cmap=cmap,norm=TwoSlopeNorm(vcenter=0,vmin=min(-3,np.nanmin(arr)),vmax=max(3,np.nanmax(arr))))
        else: im=ax.imshow(arr,aspect='auto',cmap=cmap,vmin=0)
        ax.set_xticks(range(3),domains,rotation=35,ha='right'); ax.set_title(title,fontweight='bold',fontsize=9.0); panel_label(ax,chr(65+i),x=-.12,y=1.03); fig.colorbar(im,ax=ax,fraction=.05,pad=.03)
    axs[0].set_yticks(range(len(vars))); axs[0].set_yticklabels(vars)
    for ax in axs[1:]: ax.tick_params(axis='y',labelleft=False)
    fig.subplots_adjust(wspace=.28,bottom=.19)
    save(fig,out,'Figure_S2')

def figureS3(out):
    d=read('FigureS03_ExternalAgreement.csv'); labels=[f"{r.Domain}, {r.Endpoint}" for _,r in d.iterrows()]; y=np.arange(len(d))[::-1]
    fig,axs=plt.subplots(1,3,figsize=(10.2,4.6),sharey=True)
    specs=[('Calibration_slope','Slope_CI95_lower','Slope_CI95_upper','Calibration slope',1),('Calibration_intercept','Intercept_CI95_lower','Intercept_CI95_upper','Calibration intercept, g/d',0),('CCC','CCC_CI95_lower','CCC_CI95_upper','Concordance correlation coefficient',1)]
    colors=[COLORS['blue']]*2+[COLORS['purple']]*2+[COLORS['green']]*2
    for j,(val,lo,hi,title,ref) in enumerate(specs):
        ax=axs[j]
        for yi,(_,r),cc in zip(y,d.iterrows(),colors): ax.errorbar(r[val],yi,xerr=[[r[val]-r[lo]],[r[hi]-r[val]]],fmt='o',ms=4,color=cc,ecolor=cc,capsize=0)
        ax.axvline(ref,color='#999',ls=':',lw=.8); ax.set_title(title,fontweight='bold'); panel_label(ax,chr(65+j)); style_ax(ax)
    axs[0].set_yticks(y); axs[0].set_yticklabels(labels)
    axs[1].tick_params(axis='y',labelleft=False); axs[2].tick_params(axis='y',labelleft=False); save(fig,out,'Figure_S3')

def figureS4(out):
    d=read('FigureS04_preupdate_ingredient_heterogeneity.csv'); order=d[d.Endpoint=='FN']['Ingredient class'].tolist(); fig,axs=plt.subplots(1,2,figsize=(9.2,5.0),sharey=True)
    for j,ep in enumerate(['FN','UN']):
        ax=axs[j]; g=d[d.Endpoint==ep].set_index('Ingredient class').loc[order].reset_index(); y=np.arange(len(g))[::-1]
        ax.hlines(y,0,g['RMSE g/d'],color='#9AC4DD',lw=1.1); ax.plot(g['RMSE g/d'],y,'o',color=COLORS['blue'],ms=5); ax.set_title('Fecal nitrogen (FN)' if ep=='FN' else 'Urinary nitrogen (UN)',fontweight='bold'); ax.set_xlabel('Pre-update RMSE, g/d'); style_ax(ax); panel_label(ax,chr(65+j))
    yy=np.arange(len(order))[::-1]; axs[0].set_yticks(yy); axs[0].set_yticklabels([f"{x} (n={int(d[(d.Endpoint=='FN')&(d['Ingredient class']==x)]['n source-specific diets'].iloc[0])})" for x in order]); axs[1].tick_params(axis='y',labelleft=False); save(fig,out,'Figure_S4')

def selection_grid(out):
    d=read('FigureS05_Selection_AllBudgets.csv'); fig,axs=plt.subplots(4,3,figsize=(9.4,10.4),sharex=True,sharey=True)
    strategies=['P2-PAM representative sampling (k-medoids)','Response-surface-guided sampling','Random sampling']; titles=['P2-PAM representative sampling','Response-surface-guided','Random']
    for i,b in enumerate([3,6,9,12]):
        for j,(st,title) in enumerate(zip(strategies,titles)):
            ax=axs[i,j]; g=d[(d.measured_source_specific_diets==b)&(d.sampling_strategy==st)]; base=g.drop_duplicates('source_specific_diet_id')
            for ing,gg in base.groupby('ingredient_class'): ax.scatter(gg.PC1,gg.PC2,s=11,color=INGREDIENT_COLORS.get(ing,'#BBB'),alpha=.30,edgecolors='none')
            if st=='Random sampling':
                sizes=12+220*g.selected_or_selection_frequency.clip(0,.2); ax.scatter(g.PC1,g.PC2,marker='D',s=sizes,facecolors='none',edgecolors=COLORS['orange'],lw=.55)
            else:
                sel=g[g.selected_or_selection_frequency>0]; marker='^' if st.startswith('P2-PAM') else 's'; color=COLORS['blue'] if st.startswith('P2-PAM') else COLORS['purple']; ax.scatter(sel.PC1,sel.PC2,s=25,marker=marker,color=color,edgecolors='white',lw=.3)
            if i==0: ax.set_title(title,fontweight='bold',fontsize=8.7)
            if j==0: ax.set_ylabel(f'{b} diets\nPC2');
            if i==3: ax.set_xlabel('PC1')
            style_ax(ax,False)
        panel_label(axs[i,0],chr(65+i),x=-.25,y=1.12)
    save(fig,out,'Figure_S5')

def curves3(out,source,stem,mode):
    d=read(source); fig,axs=plt.subplots(1,3,figsize=(10.2,4.0))
    metrics=[('Mean-NRMSE','Mean NRMSE'),('FN R²','FN R²'),('UN R²','UN R²')]
    if mode=='sampling':
        styles={'Response-surface-guided sampling':(COLORS['purple'],'o','RSGS'),'P2-PAM representative sampling (k-medoids)':(COLORS['blue'],'o','P2-PAM representative sampling'),'Random sampling':(COLORS['gray'],'o','Random')}
        for j,(col,title) in enumerate(metrics):
            ax=axs[j]
            for name,(c,m,l) in styles.items():
                g=d[d['Measurement-selection strategy']==name].sort_values('Budget'); ax.plot(g.Budget,g[col],marker=m,color=c,lw=1.5,label=l)
            ax.set_title(title,fontweight='bold'); ax.set_xticks([3,6,9,12]); ax.set_xlabel('Measured source-specific diets'); style_ax(ax); panel_label(ax,chr(65+j));
        axs[0].legend(frameon=False)
    else:
        tmp=d.copy(); tmp['Method']=tmp['Updating method / boundary'].map(METHOD_NAME_MAP)
        for j,(col,title) in enumerate(metrics):
            ax=axs[j]
            for m in METHOD_STYLE:
                g=tmp[tmp.Method==m].sort_values('Budget');
                if len(g)==0: continue
                c,mk,ls=METHOD_STYLE[m]; ax.plot(g.Budget,g[col],marker=mk,color=c,ls=ls,lw=1.25,label=m)
            ax.set_title(title,fontweight='bold'); ax.set_xticks([3,6,9,12]); ax.set_xlabel('Measured source-specific diets'); style_ax(ax); panel_label(ax,chr(65+j));
        fig.legend(handles=[Line2D([0],[0],color=METHOD_STYLE[m][0],ls=METHOD_STYLE[m][2],marker='o',label=m) for m in METHOD_STYLE],loc='lower center',ncol=4,frameon=False); fig.subplots_adjust(bottom=.24)
    save(fig,out,stem)

def figureS8(out):
    d=pd.read_csv(SRC/'FigureS08_MeasurementDepth.csv',skiprows=3)
    for cc in d.columns:
        if cc != 'Interval interpretation':
            d[cc]=pd.to_numeric(d[cc],errors='coerce')
    fig,axs=plt.subplots(1,3,figsize=(10.2,4.2)); metrics=[('Mean NRMSE','Mean NRMSE','Mean NRMSE allocation 2.5%','Mean NRMSE allocation 97.5%'),('FN R²','FN R²','FN allocation 2.5%','FN allocation 97.5%'),('UN R²','UN R²','UN allocation 2.5%','UN allocation 97.5%')]
    colors={1:COLORS['blue'],3:COLORS['orange'],6:COLORS['green']}; marks={1:'o',3:'s',6:'^'}
    for j,(col,title,lo,hi) in enumerate(metrics):
        ax=axs[j]
        for rep in [1,3,6]:
            g=d[d['Repeats per selected source']==rep].sort_values('Measured source-specific diets'); ax.plot(g['Measured source-specific diets'],g[col],marker=marks[rep],color=colors[rep],lw=1.4,label=f'{rep} pig per diet' if rep==1 else f'{rep} pigs per diet')
            if rep==1: ax.fill_between(g['Measured source-specific diets'],g[lo],g[hi],color='#DDEBF3',alpha=1.0,zorder=0)
        ax.set_title(title,fontweight='bold'); ax.set_xticks([3,6,9,12]); ax.set_xlabel('Measured source-specific diets'); style_ax(ax); panel_label(ax,chr(65+j))
    axs[0].set_ylabel('Mean NRMSE'); axs[1].set_ylabel('R²'); axs[2].set_ylabel('R²')
    handles=[Line2D([0],[0],color=colors[r],marker=marks[r],label=(f'{r} pig per diet' if r==1 else f'{r} pigs per diet')) for r in [1,3,6]]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False); fig.subplots_adjust(bottom=.2); save(fig,out,'Figure_S8')

def figureS9(out):
    d=read('FigureS09_S10_full_matrix_summary.csv'); d=d[d.Method=='CG-HPP']; fig,axs=plt.subplots(1,3,figsize=(10.4,4.2)); specs=[('Primary_mean','Mean NRMSE'),('FN_R2_mean','FN R²'),('UN_R2_mean','UN R²')]
    for j,(col,title) in enumerate(specs):
        ax=axs[j]; ax.axvspan(12.01,21.7,color='#F0F1F2'); ax.axvline(12,color='#BBB',ls=':',lw=.7)
        for m in ['MLR','RF','TabM','MLR+RF']:
            g=d[d.Architecture==m].sort_values('Budget'); ax.plot(g.Budget,g[col],marker=MODEL_MARKERS[m],color=MODEL_COLORS[m],lw=1.4,label=m)
        ax.set_title(title,fontweight='bold'); ax.set_xticks([3,6,9,12,15,18,21]); ax.set_xlabel('Measured source-specific diets'); style_ax(ax); panel_label(ax,chr(65+j))
    axs[0].legend(frameon=False); save(fig,out,'Figure_S9')

def figureS10(out):
    d=read('FigureS09_S10_full_matrix_summary.csv'); fig,axs=plt.subplots(2,2,figsize=(8.8,7.2),sharex=True,sharey=True)
    for i,(ax,arch) in enumerate(zip(axs.flat,['MLR','RF','TabM','MLR+RF'])):
        ax.axvspan(12.01,21.7,color='#F0F1F2')
        for m in METHOD_STYLE:
            g=d[(d.Architecture==arch)&(d.Method==m)].sort_values('Budget'); c,mk,ls=METHOD_STYLE[m]; ax.plot(g.Budget,g.Primary_mean,marker=mk,color=c,ls=ls,lw=1.2,ms=3.5,label=m)
        ax.set_title(arch,fontweight='bold',loc='left'); ax.set_xticks([3,6,9,12,15,18,21]); style_ax(ax); panel_label(ax,chr(65+i),x=-.12,y=1.04)
    axs[1,0].set_xlabel('Measured source-specific diets'); axs[1,1].set_xlabel('Measured source-specific diets'); axs[0,0].set_ylabel('Mean NRMSE'); axs[1,0].set_ylabel('Mean NRMSE')
    fig.legend(handles=[Line2D([0],[0],color=METHOD_STYLE[m][0],ls=METHOD_STYLE[m][2],marker='o',label=m) for m in METHOD_STYLE],loc='lower center',ncol=4,frameon=False); fig.subplots_adjust(bottom=.18); save(fig,out,'Figure_S10')

FIGURE_TARGETS = [
    'Figure_1','Figure_2','Figure_3','Figure_4','Figure_5',
    'Figure_S1','Figure_S2','Figure_S3','Figure_S4','Figure_S5',
    'Figure_S6','Figure_S7','Figure_S8','Figure_S9','Figure_S10'
]

def build_named(name:str, outdir:str|Path):
    """Build one figure in an isolated worker-safe call."""
    out=Path(outdir)
    dispatch={
        'Figure_1':figure1,'Figure_2':figure2,'Figure_3':figure3,'Figure_4':figure4,'Figure_5':figure5,
        'Figure_S1':figureS1,'Figure_S2':figureS2,'Figure_S3':figureS3,'Figure_S4':figureS4,
        'Figure_S5':selection_grid,'Figure_S8':figureS8,'Figure_S9':figureS9,'Figure_S10':figureS10,
    }
    if name=='Figure_S6': curves3(out,'FigureS06_SamplingCurves.csv','Figure_S6','sampling')
    elif name=='Figure_S7': curves3(out,'FigureS07_UpdatingCurves.csv','Figure_S7','updating')
    elif name in dispatch: dispatch[name](out)
    else: raise ValueError(f'Unknown figure target: {name}')
    return name

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,default=ROOT/'generated/rebuilt_figures')
    ap.add_argument('--jobs',type=int,default=1,help='Parallel isolated figure workers (default: 1).')
    ap.add_argument('--only',choices=FIGURE_TARGETS,help='Build a single figure target.')
    args=ap.parse_args(); out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    targets=[args.only] if args.only else FIGURE_TARGETS
    if len(targets)==1 or args.jobs==1:
        for name in targets:
            build_named(name,out); print('built',name,flush=True)
    else:
        # Process isolation prevents Matplotlib backend/font state from accumulating across
        # the 15 sequential EPS exports and makes the publication build robust across clean environments.
        with ProcessPoolExecutor(max_workers=max(1,args.jobs)) as ex:
            futs={ex.submit(build_named,name,str(out)):name for name in targets}
            for fut in as_completed(futs):
                name=futs[fut]; fut.result(); print('built',name,flush=True)
    png=sorted(out.glob('*.png')); eps=sorted(out.glob('*.eps'))
    expected=len(targets)
    if args.only:
        if not (out/f'{args.only}.png').exists() or not (out/f'{args.only}.eps').exists(): raise AssertionError(args.only)
    elif len(png)!=15 or len(eps)!=15: raise AssertionError((len(png),len(eps)))
    print(f'PASS: {expected} figure target(s) regenerated from packaged figure-source tables in PNG/EPS formats')

if __name__=='__main__': main()
