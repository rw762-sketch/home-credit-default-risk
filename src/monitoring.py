"""Development-selected cutoff, calibration diagnostics, and distribution drift."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, log_loss, precision_recall_curve,
                             confusion_matrix)
from src.artifacts import artifact_path


def threshold_sweep(y, probability, beta=2):
    """Evaluate every distinct score with the prediction rule p >= threshold."""
    precision, recall, thresholds = precision_recall_curve(y, probability)
    precision, recall = precision[:-1], recall[:-1]
    denom = beta**2 * precision + recall
    score = np.divide((1+beta**2)*precision*recall, denom,
                      out=np.zeros_like(denom), where=denom>0)
    table = pd.DataFrame({'threshold': thresholds, 'precision': precision,
                          'recall': recall, 'f2': score})
    # Highest threshold wins an exact tie, reducing the number flagged.
    best = table.sort_values(['f2', 'threshold'], ascending=False).iloc[0]
    return table, float(best.threshold)


def cutoff_metrics(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, np.asarray(p)>=threshold,
                                     labels=[0, 1]).ravel()
    precision = tp/(tp+fp) if tp+fp else 0.
    recall = tp/(tp+fn) if tp+fn else 0.
    return {'threshold': float(threshold), 'precision': precision, 'recall': recall,
            'f2': 5*precision*recall/(4*precision+recall) if precision+recall else 0.,
            'flagged_fraction': float((tp+fp)/len(y)),
            'true_positive': int(tp), 'false_positive': int(fp),
            'false_negative': int(fn), 'true_negative': int(tn)}


def calibration_table(y, p, bins=10):
    frame = pd.DataFrame({'target':y, 'probability':p})
    frame['bin'] = pd.qcut(frame.probability, bins, duplicates='drop')
    if frame['bin'].isna().all():
        frame['bin'] = 0
    table = frame.groupby('bin', observed=True).agg(
        count=('target','size'), observed_rate=('target','mean'),
        mean_probability=('probability','mean')).reset_index()
    table['absolute_gap'] = abs(table.observed_rate-table.mean_probability)
    return table


def psi(reference, current, bins=10, simulations=250, seed=42):
    """PSI with reference-fixed bins, missing bucket, and finite-sample null."""
    ref, cur = pd.Series(reference), pd.Series(current)
    if len(ref)==0 or len(cur)==0:
        raise ValueError('PSI requires two nonempty samples.')
    if pd.api.types.is_numeric_dtype(ref) and ref.nunique()>bins:
        values = ref.dropna().to_numpy()
        internal = np.unique(np.quantile(values, np.linspace(0,1,bins+1)[1:-1]))
        edges = np.r_[-np.inf,internal,np.inf]
        # Final bucket is always reserved for missing values.
        r = np.where(ref.isna(),len(edges)-1,np.searchsorted(edges,ref,side='right')-1)
        c = np.where(cur.isna(),len(edges)-1,np.searchsorted(edges,cur,side='right')-1)
        size = len(edges)
    else:
        keys = ref.dropna().unique().tolist()
        codes = {key:i for i,key in enumerate(keys)}
        # Include explicit unseen and missing buckets.
        r = ref.map(codes).fillna(len(keys)).astype(int).to_numpy(copy=True)
        c = cur.map(codes).fillna(len(keys)).astype(int).to_numpy(copy=True)
        r[ref.isna()] = len(keys)+1
        c[cur.isna()] = len(keys)+1
        size = len(keys)+2
    rc, cc = np.bincount(r,minlength=size), np.bincount(c,minlength=size)
    def statistic(a,b):
        # Jeffreys pseudocount prevents infinite values for empty buckets.
        a=(a+.5)/(a.sum(axis=-1,keepdims=True)+.5*size)
        b=(b+.5)/(b.sum(axis=-1,keepdims=True)+.5*size)
        return np.sum((b-a)*np.log(b/a),axis=-1)
    value = float(statistic(rc,cc))
    rng=np.random.default_rng(seed)
    null_probability=rc/rc.sum()
    # Simulate both sample sizes under the fixed reference binning.
    null=statistic(rng.multinomial(len(ref),null_probability,size=simulations),
                   rng.multinomial(len(cur),null_probability,size=simulations))
    return {'psi':value,'null_p95':float(np.quantile(null,.95)),
            'exceeds_null_p95':bool(value>np.quantile(null,.95)),
            'buckets':size,'reference_rows':len(ref),'current_rows':len(cur),
            'missing_reference':float(ref.isna().mean()),
            'missing_current':float(cur.isna().mean())}


def save_fig(out,name):
    plt.tight_layout()
    plt.savefig(artifact_path(out,name),dpi=170,bbox_inches='tight')
    plt.close()


def analyse(out, dev_frame, hold_frame, oof, hold_probability, test_frame=None,
            test_probability=None, reference_scores=None):
    sweep, threshold = threshold_sweep(dev_frame.TARGET, oof)
    sweep.to_csv(artifact_path(out,'threshold_sweep.csv'),index=False)
    rows=[]
    for split,y,p in [('development OOF',dev_frame.TARGET,oof),
                      ('previously inspected holdout',hold_frame.TARGET,hold_probability)]:
        for policy,t in [('fixed 0.5',.5),('development-selected F2',threshold)]:
            rows.append({'split':split,'policy':policy,**cutoff_metrics(y,p,t)})
    comparison=pd.DataFrame(rows)
    comparison.to_csv(artifact_path(out,'threshold_comparison.csv'),index=False)
    plt.figure(figsize=(8,4))
    for name in ['precision','recall','f2']:
        plt.plot(sweep.threshold,sweep[name],label=name)
    plt.axvline(threshold,color='#147d78',linestyle='--',label=f'selected {threshold:.3f}')
    plt.xlim(0,.5);plt.ylim(0,1);plt.xlabel('Cutoff on development OOF predictions')
    plt.ylabel('Metric');plt.title('Choose the cutoff using development data');plt.legend()
    save_fig(out,'threshold_curve.png')
    cals=[];cal_metrics={}
    plt.figure(figsize=(7,4))
    plt.plot([0,1],[0,1],color='gray',linestyle='--',label='perfect calibration')
    for name,y,p in [('development OOF',dev_frame.TARGET,oof),
                      ('previously inspected holdout',hold_frame.TARGET,hold_probability)]:
        tab=calibration_table(np.asarray(y),p)
        tab['split']=name;cals.append(tab)
        cal_metrics[name]={'brier':float(brier_score_loss(y,p)),
            'log_loss':float(log_loss(y,p)),
            'mean_probability':float(np.mean(p)), 'observed_rate':float(np.mean(y)),
            'weighted_absolute_gap':float(np.average(tab.absolute_gap,weights=tab['count']))}
        plt.plot(tab.mean_probability,tab.observed_rate,'o-',label=name)
    pd.concat(cals).to_csv(artifact_path(out,'calibration.csv'),index=False)
    plt.xlabel('Mean predicted probability');plt.ylabel('Observed positive rate')
    plt.title('Calibration by score decile');plt.legend()
    save_fig(out,'calibration_curve.png')
    quality=[]
    for split,frame in [('development',dev_frame),('unlabeled test',test_frame)]:
        if frame is None:continue
        for column in frame.columns:
            if column in ['TARGET','SK_ID_CURR']:continue
            quality.append({'split':split,'feature':column,'rows':len(frame),
                'missing_fraction':float(frame[column].isna().mean()),
                'unique_nonmissing':int(frame[column].nunique()),
                'employment_sentinel_count':int((frame[column]==365243).sum())
                  if column=='DAYS_EMPLOYED' else 0})
    pd.DataFrame(quality).to_csv(artifact_path(out,'data_quality.csv'),index=False)
    if 'DAYS_BIRTH' in dev_frame:
        age=pd.DataFrame({'age_band':pd.cut(-dev_frame.DAYS_BIRTH/365.25,
            [0,30,40,50,120],labels=['under 30','30-40','40-50','50+']),
            'target':dev_frame.TARGET})
        age=age.groupby('age_band',observed=True).agg(count=('target','size'),
            positive_rate=('target','mean')).reset_index()
        age.to_csv(artifact_path(out,'age_risk.csv'),index=False)
        plt.figure(figsize=(7,3.5));plt.bar(age.age_band.astype(str),age.positive_rate,color='#147d78')
        plt.xlabel('Age band');plt.ylabel('Observed repayment-difficulty rate')
        plt.title('Development-only age summary (descriptive)');save_fig(out,'age_risk.png')
    drift=[]
    if test_frame is not None:
        for column in dev_frame.columns:
            if column in ['TARGET','SK_ID_CURR']:continue
            drift.append({'feature':column,'kind':'raw feature',
                **psi(dev_frame[column],test_frame[column])})
        # Both sets are scored by the same development-fitted model; neither was fitted.
        drift.append({'feature':'MODEL_SCORE','kind':'same-model score',
            **psi(reference_scores,test_probability)})
        pd.DataFrame(drift).sort_values('psi',ascending=False).to_csv(
            artifact_path(out,'drift.csv'),index=False)
        top=pd.DataFrame(drift).sort_values('psi').tail(12)
        plt.figure(figsize=(8,5));plt.barh(top.feature,top.psi,color='#147d78')
        plt.xlabel('PSI with reference-fixed bins');plt.title('Development features vs unlabeled test')
        save_fig(out,'drift_chart.png')
    metrics={'threshold':threshold,'threshold_objective':'maximize development OOF F2 (beta=2)',
        'tie_break':'highest threshold among equal F2 values',
        'calibration':cal_metrics,'holdout_status':'Previously inspected; follow-up diagnostic, not a fresh final test.',
        'drift':{'feature_reference':'development rows','score_reference':'reserved holdout rows',
                 'score_model':'same development-fitted evaluation model for holdout and test',
                 'null_simulations':250,'null_quantile':.95,
                 'interpretation':'Descriptive distribution comparison; feature-wise null flags are not multiplicity-corrected and do not establish performance deterioration.'},
        'references':['https://www.kaggle.com/code/truenikita/home-credit-psi-drift-and-re-setting-the-cutoff',
                      'https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction']}
    artifact_path(out,'monitoring_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    return metrics
