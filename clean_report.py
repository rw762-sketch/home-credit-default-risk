"""Build a readable report from saved predictions; never retrain a model."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from src.artifacts import artifact_path, latest_run, run_date

ROOT = Path(__file__).resolve().parent
NAMES = {'gradient_boosting': 'Histogram gradient boosting',
         'logistic_regression': 'Logistic regression', 'random_forest': 'Random forest',
         'lightgbm': 'LightGBM'}
REFERENCES = [
    ('Dataset and column dictionary', 'https://www.kaggle.com/competitions/home-credit-default-risk/data'),
    ('Presentation reference: results, approach, train/test consistency', 'https://github.com/sdolcecanto/home-credit-default-risk'),
    ('Presentation reference: model comparison and explainability', 'https://github.com/ManishThumma/credit-risk-predictor'),
    ('scikit-learn: preprocessing and leakage', 'https://scikit-learn.org/stable/common_pitfalls.html'),
    ('Nikita Ageev: PSI, drift, and cutoff diagnostics', 'https://www.kaggle.com/code/truenikita/home-credit-psi-drift-and-re-setting-the-cutoff'),
    ('William Koehrsen: EDA, anomalies, and baseline modeling', 'https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction'),
    ('Home Aloan: first-place technical solution and experiment narrative', 'https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution'),
]


def build(source, destination):
    source, destination = source.resolve(), destination.resolve()
    metrics = json.loads((artifact_path(source, 'metrics.json')).read_text())
    if metrics['data_source'] == 'SYNTHETIC DEMO':
        raise ValueError('This report requires real Home Credit results.')
    comparison = pd.read_csv(artifact_path(source, 'model_comparison.csv'))
    hold = pd.read_csv(artifact_path(source, 'holdout_predictions.csv'))
    y, p = hold.TARGET.to_numpy(), hold.probability.to_numpy()
    if len(hold) != metrics['holdout_rows'] or not hold.SK_ID_CURR.is_unique:
        raise ValueError('Holdout row count or applicant IDs are inconsistent.')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid probabilities.')
    if not np.isclose(roc_auc_score(y, p), metrics['holdout_auc']):
        raise ValueError('Saved AUC does not reconcile with predictions.')
    if not np.isclose(average_precision_score(y, p), metrics['holdout_average_precision']):
        raise ValueError('Saved average precision does not reconcile.')
    pred = (p >= metrics['threshold']).astype(int)
    if not np.array_equal(pred, hold.predicted.to_numpy()):
        raise ValueError('Saved class predictions do not match the recorded threshold.')
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    now = datetime.now(ZoneInfo('America/New_York'))
    run_date = datetime.fromtimestamp((artifact_path(source, 'metrics.json')).stat().st_mtime,
                                      ZoneInfo('America/New_York')).strftime('%B %d, %Y')
    for folder in ['figures', 'data']:
        (destination / folder).mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                        'axes.spines.right': False, 'figure.facecolor': 'white'})
    def save(fig, name):
        fig.savefig(artifact_path(destination / 'figures', name), dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ordered = comparison.sort_values('cv_auc_mean')
    ax.barh([NAMES.get(n, n) for n in ordered.model], ordered.cv_auc_mean,
            xerr=ordered.cv_auc_std,
            color=['#147d78' if n == metrics['selected_model'] else '#b5c3cb'
                   for n in ordered.model], capsize=4)
    ax.set_xlim(0.65, 0.80)
    ax.set_xlabel('Mean five-fold development ROC-AUC (bars: fold standard deviation)')
    for i, value in enumerate(ordered.cv_auc_mean):
        ax.text(value + .006, i, f'{value:.4f}', va='center')
    fig.tight_layout()
    save(fig, 'model_comparison.png')
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    counts = np.array([[tn, fp], [fn, tp]])
    ax.imshow(counts, cmap='Blues')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{counts[i,j]:,}', ha='center', va='center', fontsize=17,
                    color='white' if counts[i,j] > counts.max()/2 else '#142a38')
    ax.set_xticks([0, 1], ['Predicted 0', 'Predicted 1'])
    ax.set_yticks([0, 1], ['Actual 0', 'Actual 1'])
    ax.set_title(f'Holdout confusion matrix | cutoff {metrics["threshold"]:g}')
    save(fig, 'confusion_matrix.png')
    for name in ['roc_curve.png', 'precision_recall_curve.png', 'shap_summary.png',
                 'target_distribution.png', 'correlation_heatmap.png', 'feature_importance.png',
                 'threshold_curve.png', 'calibration_curve.png', 'drift_chart.png', 'age_risk.png']:
        if (artifact_path(source, name)).exists():
            shutil.copy2(artifact_path(source, name), artifact_path(destination / 'figures', name))
    for name in ['metrics.json', 'model_comparison.csv', 'feature_importance.csv', 'missing_values.csv',
                 'threshold_comparison.csv', 'calibration.csv', 'drift.csv', 'monitoring_metrics.json',
                 'data_quality.csv', 'age_risk.csv']:
        if artifact_path(source, name).exists():
            shutil.copy2(artifact_path(source, name), artifact_path(destination / 'data', name))
    prevalence = float(y.mean())
    ntest = len(pd.read_csv(artifact_path(source, 'submission.csv'))) if (artifact_path(source, 'submission.csv')).exists() else 0
    sections = []
    def section(title, blocks):
        sections.append((title, blocks))
    def text(value):
        return ('text', value)
    def table(headers, rows):
        return ('table', (headers, rows))
    def figure(name, caption, height=220):
        return ('figure', (name, caption, height))
    section('Home Credit: credit-risk modeling', [
        text('Research question: how well can application-time attributes rank repayment '
             'difficulty, and how does the classification cutoff change the errors? '
             'This study compares three classifiers, then examines probability accuracy '
             'and distribution differences between development and competition test data.'),
        text(f'The selected {NAMES[metrics["selected_model"]].lower()} model achieved '
             f'holdout ROC-AUC {metrics["holdout_auc"]:.4f} after selection on development '
             'cross-validation. At cutoff 0.5, recall is '
             f'{metrics["recall"]:.2%}; ranking performance alone does not establish '
             'a useful classification rule.'),
        table(['Measure', 'Saved result'], [
            ['Labeled applications', f'{metrics["development_rows"]+metrics["holdout_rows"]:,}'],
            ['Development / reserved holdout', f'{metrics["development_rows"]:,} / {len(hold):,}'],
            ['Holdout ROC-AUC', f'{metrics["holdout_auc"]:.4f}'],
            ['Holdout average precision', f'{metrics["holdout_average_precision"]:.4f}'],
            ['Precision / recall at cutoff 0.5', f'{metrics["precision"]:.2%} / {metrics["recall"]:.2%}'],
            ['Local unlabeled test predictions', f'{ntest:,}'],
        ]),
        text('Objective: predict TARGET = 1, which denotes repayment difficulty in the competition. '
             'TARGET = 0 denotes the other class. Applicant ID is excluded from predictors. '
             'The Kaggle test table has no labels: it supplies predictions, not a measured AUC.'),
        text(f'Experiment date: {run_date}. Presentation updated: {now:%Y-%m-%d %H:%M %Z}. '
             'Results are read from saved artifacts. The random holdout has been inspected '
             'and reused for follow-up diagnostics. No Kaggle leaderboard score or '
             'temporal/external validation result is available.'),
    ])
    section('Data and experimental design', [
        text('The public Home Credit competition contains application records linked to credit '
             'and repayment histories. This experiment uses the main application table only: '
             '307,511 labeled rows and 122 input columns, including the target and applicant ID. '
             'Historical tables have not been incorporated. The positive class denotes '
             'repayment difficulty, rather than a complete measure of loan profitability.'),
        table(['Stage', 'Protocol'], [
            ['Split', f'Seed 42; stratified 80/20 split: {metrics["development_rows"]:,} development rows and {len(hold):,} holdout rows.'],
            ['Features', 'Exclude ID and target; replace employment sentinel 365243; add credit/income, annuity/income, and credit/annuity ratios.'],
            ['Preprocessing', 'Learn numeric median imputation, scaling, categorical imputation, and one-hot categories within each training fold. Rare-category minimum count: 20.'],
            ['Model selection', 'Five stratified development folds shared across all candidates; select the highest mean fold ROC-AUC.'],
            ['Evaluation', 'Fit the winner on development rows; score the reserved holdout. In the extension, select F2 cutoff on development OOF predictions and diagnose the reused holdout.'],
        ]),
        text('Out-of-fold (OOF) means each development row is scored by a fold model that '
             'did not fit it. The same folds are used for model and cutoff selection, '
             'so optimized development metrics may be optimistic. Applicant IDs are '
             'checked for uniqueness, but this is not a verified customer-group or time split.'),
        text('Configurations: logistic regression uses max_iter=2000; random forest uses '
             '100 trees, depth 12, and minimum leaf size 10; histogram gradient boosting '
             'uses 150 iterations, 15 leaf nodes, and L2 regularization 1. '
             'No hyperparameter search or feature-family ablation was performed.'),
    ])
    section('Experiment 1: compare model families', [
        text('Hypothesis: nonlinear models can capture relationships beyond the linear '
             'baseline. All candidates use the same application features and development '
             'folds, so the comparison isolates the configured model family.'),
        table(['Model', 'CV mean AUC', 'Fold SD'], [
            [NAMES.get(row.model, row.model), f'{row.cv_auc_mean:.4f}', f'{row.cv_auc_std:.4f}']
            for row in comparison.itertuples()]),
        figure('model_comparison.png', 'All candidates use the same five development folds. Fold SD describes variation; it is not a confidence interval.', 130),
        figure('roc_curve.png', f'Only the CV winner is evaluated on the {len(hold):,}-row holdout. ROC-AUC measures ranking across cutoffs.', 180),
        text('The winner is chosen by mean fold ROC-AUC. The logistic and forest values above are '
             'development scores; this run does not report their holdout scores. Model selection '
             'must not compare these CV values as if they were Kaggle leaderboard results.'),
        text('Decision: retain the development-CV winner for the remaining diagnostics. '
             'This comparison does not show that tuned forests or other boosting configurations '
             'would perform worse; only the listed configurations were evaluated.'),
    ])
    section('Baseline decision rule: cutoff 0.5', [
        text(f'The holdout contains {int(y.sum()):,} positive cases ({prevalence:.2%}). '
             f'At cutoff 0.5, {tp:,} are caught and {fn:,} are missed. '
             f'There are {fp:,} false alarms among {int((y==0).sum()):,} negative cases.'),
        figure('confusion_matrix.png', 'The fixed 0.5 cutoff yields high precision among the few flagged cases, but very low recall.', 165),
        figure('precision_recall_curve.png', f'Average precision is {metrics["holdout_average_precision"]:.4f}; the positive-class prevalence is {prevalence:.4f}.', 215),
        text('Precision asks: of the cases flagged, how many are positive? Recall asks: of all '
             'positive cases, how many were caught? AUC does not choose a cutoff and does not '
             'guarantee calibrated probabilities. Choose a future cutoff using development '
             'out-of-fold predictions and an explicit objective, then use fresh evaluation data.'),
    ])
    feature_blocks = [text('The explanation below uses 100 development applicants. SHAP describes '
        'how the fitted model uses features; it does not identify causes of repayment difficulty.')]
    if (artifact_path(destination / 'figures', 'shap_summary.png')).exists():
        feature_blocks.append(figure('shap_summary.png', 'Higher external scores tend to push the model output down in this sample. SHAP values are on the model output scale, not percentage-point changes in default probability.', 440))
    feature_blocks.append(text('The external-score fields dominate this small explanation sample. '
        'The supplied dictionary identifies them only as normalized scores from external data sources. '
        'Their provenance and application-time availability still need review. Permutation importance '
        'is computed on 500 development rows in sample and can understate correlated predictors.'))
    section('Interpretation: features used by the model', feature_blocks)
    monitoring_file = artifact_path(source, 'monitoring_metrics.json')
    has_monitoring = monitoring_file.exists()
    if has_monitoring:
        monitor = json.loads(monitoring_file.read_text())
        cutoff = pd.read_csv(artifact_path(source, 'threshold_comparison.csv'))
        hcut = cutoff[cutoff.split == 'previously inspected holdout']
        selected = hcut[hcut.policy == 'development-selected F2'].iloc[0]
        sections[0][1].insert(2, text(
            f'Monitoring extension: development F2 selects cutoff {monitor["threshold"]:.4f}. '
            f'On the reused holdout, recall is {selected.recall:.2%} and precision is '
            f'{selected.precision:.2%}. It catches {int(selected.true_positive):,} positive '
            f'cases with {int(selected.false_positive):,} false positives. The AUC and model '
            'are unchanged; this is a cutoff tradeoff, not fresh final validation.'))
        section('Experiment 2: select the decision cutoff', [
            text(f'The selected cutoff is {monitor["threshold"]:.6f}, chosen by maximizing '
                 'development out-of-fold F2. F2 gives recall more weight than precision. '
                 'This objective was chosen before running the extension; no holdout labels '
                 'are used in the cutoff search. Exact ties select the highest cutoff.'),
            table(['Holdout policy', 'Cutoff', 'Precision', 'Recall'], [
                ['Fixed 0.5' if r.policy=='fixed 0.5' else 'Development F2',
                 f'{r.threshold:.4f}', f'{r.precision:.2%}', f'{r.recall:.2%}']
                for r in hcut.itertuples()]),
            figure('threshold_curve.png', 'The search uses development OOF predictions. The final point of the precision-recall curve has no threshold and is excluded.', 260),
            text(f'The selected cutoff catches {int(selected.true_positive):,} positive holdout '
                 f'cases, misses {int(selected.false_negative):,}, and flags '
                 f'{int(selected.false_positive):,} negative cases. More recall comes with '
                 'more false positives. A lower cutoff does not improve ROC-AUC or calibration.'),
            text('The holdout was already inspected in the baseline project. These are follow-up '
                 'diagnostics on reused evaluation data, not fresh final validation. F2 is '
                 'a research objective and does not define a lending policy or expected profit.'),
        ])
        section('Diagnostic 1: probability calibration', [
            text('Calibration compares predicted probabilities with observed positive rates. '
                 'These diagnostics do not fit a recalibration model. Equal-frequency score '
                 'bins are descriptive; their boundaries are estimated separately for each split.'),
            table(['Split', 'Brier score', 'Log loss'], [
                ['Development OOF' if 'development' in name else 'Reused holdout',
                 f'{value["brier"]:.5f}', f'{value["log_loss"]:.5f}']
                for name,value in monitor['calibration'].items()]),
            figure('calibration_curve.png', 'Lower Brier score and log loss indicate better probability accuracy. Compare curves with the diagonal; they are not proof of future calibration.', 285),
            text('Each development probability was produced by a model that did not fit that '
                 'row. However, the same development folds select the model and the cutoff, '
                 'so their optimized results can be optimistic. They are development '
                 'estimates, not a substitute for independent evaluation.'),
        ])
        drift_file = artifact_path(source, 'drift.csv')
        if drift_file.exists():
            drift = pd.read_csv(drift_file)
            top = drift[drift.kind=='raw feature'].head(4)
            score = drift[drift.feature=='MODEL_SCORE'].iloc[0]
            section('Diagnostic 2: distribution drift', [
                text(f'Same-model score PSI is {score.psi:.5f}. Both holdout and unlabeled '
                     'test scores come from the same development-fitted model. Raw feature '
                     'PSI instead compares development rows with the unlabeled test table.'),
                table(['Largest raw-feature shifts', 'PSI', 'Null 95%'], [
                    [r.feature,f'{r.psi:.4f}',f'{r.null_p95:.4f}'] for r in top.itertuples()]),
                figure('drift_chart.png', 'Reference-fixed quantile/category buckets include missing and unseen categories. A 0.5 pseudocount handles empty buckets.', 245),
                text('For each feature, 250 multinomial draws estimate the no-change PSI '
                     'distribution at these sample sizes, conditional on the reference bins. '
                     'The 95% null flags are exploratory and are not corrected for multiple '
                     'comparisons. A flagged distribution is a prompt to inspect the data, '
                     'not proof of deteriorating predictions. No test labels are available '
                     'to assess its calibration or AUC.'),
                text('The gentle introduction informed the development-only age summary, '
                     'missing-value audit, and employment-sentinel counts. These are saved '
                     'as numbered tables and an age chart. They describe associations, '
                     'not causal effects. Drift/calibration diagnostics follow the '
                     'questions raised in Ageev\'s notebook; no synthetic profit claims '
                     'or automatic 0.1/0.25 PSI decision rule is adopted.'),
            ])
    section('Limitations and research priorities', [
        table(['Priority', 'Question to test'], [
            ['Historical features', 'Does one family of applicant-linked repayment aggregates improve development AUC on unchanged folds?'],
            ['Alternative models', 'Does LightGBM add predictive value under the same features and evaluation protocol?'],
            ['Probability and cutoff', 'Does fold-fitted recalibration help probability accuracy? How does the error tradeoff change under a different stated objective?'],
            ['Independent evaluation', 'Can the frozen pipeline be evaluated on genuinely new labeled data with a defensible time or customer boundary?'],
        ]),
        text('Scope: the application table only. Secondary-table aggregation, hyperparameter search, '
             'and fitted probability recalibration have not been performed. '
             + ('Development-based threshold selection and calibration/drift diagnostics are implemented. ' if has_monitoring else 'Development-based threshold selection and calibration/drift diagnostics remain future work. ') +
             'The split is random, not temporal or external. The holdout has now been inspected; '
             'repeated iteration against it cannot establish an untouched final test result.'),
        text('Next experiments: compare LightGBM and historical-table features on development '
             'folds, evaluate alternative cutoff objectives and fitted recalibration, '
             'and use fresh data for final performance assessment. '
             'No leaderboard score, lending decision rule, or financial impact is established.'),
        text('GitHub presentation references were reviewed for organization and documentation only. '
             'External implementations use different features and evaluation designs, so their '
             'reported scores are not controlled comparisons with this run.'),
        *[('link', item) for item in REFERENCES],
    ])
    md, body = [], []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ReportBody', fontName='Helvetica', fontSize=10,
        leading=14, textColor=colors.HexColor('#263c49'), spaceAfter=11))
    styles.add(ParagraphStyle(name='CaptionClean', fontSize=8, leading=11,
        textColor=colors.HexColor('#526675'), spaceAfter=11))
    styles['Heading1'].fontSize, styles['Heading1'].leading = 23, 27
    styles['Heading1'].textColor = colors.HexColor('#123c4a')
    story = []
    for index, (title, blocks) in enumerate(sections):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(html.escape(title), styles['Heading1']))
        story.append(Spacer(1, 14))
        md.append('# ' + title if not index else '## ' + title)
        body.append(f'<section><h{"1" if not index else "2"}>{html.escape(title)}</h{"1" if not index else "2"}>')
        for kind, value in blocks:
            if kind == 'text':
                md.append(value)
                body.append('<p>' + html.escape(value) + '</p>')
                story.append(Paragraph(html.escape(value), styles['ReportBody']))
            elif kind == 'table':
                headers, rows = value
                md.append('\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers))+' |',
                           *['| ' + ' | '.join(row) + ' |' for row in rows]]))
                body.append('<div class="table-wrap"><table><thead><tr>' + ''.join(f'<th>{html.escape(c)}</th>' for c in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(c)}</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table></div>')
                widths = {2:[170,334],3:[300,102,102],4:[210,98,98,98]}[len(headers)]
                cells = [[Paragraph(html.escape(c), styles['ReportBody']) for c in row] for row in [headers]+rows]
                t = Table(cells, colWidths=widths, hAlign='LEFT')
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e1efed')),
                    ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),9),
                    ('RIGHTPADDING',(0,0),(-1,-1),9),('TOPPADDING',(0,0),(-1,-1),6),
                    ('BOTTOMPADDING',(0,0),(-1,-1),6),
                    ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#d4e0e5'))]))
                story.extend([t, Spacer(1, 15)])
            elif kind == 'figure':
                name, caption, max_height = value
                rel = 'figures/' + artifact_path(destination / 'figures', name).name
                md.append(f'![{caption}]({rel})')
                body.append(f'<figure><img src="{rel}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
                iw, ih = ImageReader(str(destination / rel)).getSize()
                scale = min(504/iw, max_height/ih)
                story.extend([Image(str(destination / rel), width=iw*scale, height=ih*scale),
                              Paragraph(html.escape(caption), styles['CaptionClean'])])
            else:
                label, url = value
                md.append(f'- [{label}]({url})')
                body.append(f'<p class="reference"><a href="{url}">{html.escape(label)}</a></p>')
                story.append(Paragraph(f'<link href="{url}" color="#147d78">{html.escape(label)}</link>', styles['CaptionClean']))
        body.append('</section>')
        md.append('')
    (artifact_path(destination, 'report.md')).write_text('\n\n'.join(md)+'\n')
    css = '''body{margin:0;background:#f2f5f6;color:#263c49;font:17px/1.65 system-ui,sans-serif}main{max-width:1000px;margin:48px auto;padding:0 24px}header{padding:28px 0;color:#147d78;font-size:13px;letter-spacing:2px;text-transform:uppercase}section{background:white;padding:36px 44px;margin-bottom:24px;border:1px solid #dce5e9;border-radius:12px}h1,h2{color:#123c4a;line-height:1.2}h1{font-size:38px}h2{font-size:27px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:12px;text-align:left;border-bottom:1px solid #dce5e9}th{background:#e1efed}figure{margin:28px 0}img{display:block;max-width:100%;max-height:850px;margin:auto}figcaption,.reference{font-size:14px;color:#526675}a{color:#147d78}.table-wrap{overflow:auto}@media(max-width:600px){main{padding:0 12px;margin:16px auto}section{padding:24px 18px}h1{font-size:30px}}@media print{body{background:white}main{margin:0;max-width:none}section{border:0;break-before:page}header{display:none}}'''
    (artifact_path(destination, 'index.html')).write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Home Credit | Model and results</title><style>'+css+'</style><main><header>Gradient boosting / Home Credit / Research results</header>'+''.join(body)+'</main></html>')
    pdf_path = artifact_path(destination, 'results.pdf')
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#d4e0e5'))
        canvas.line(54, 42, 558, 42)
        canvas.setFillColor(colors.HexColor('#526675'))
        canvas.setFont('Helvetica', 8)
        canvas.drawString(54, 28, 'HOME CREDIT | APPLICATION-TABLE BENCHMARK')
        canvas.drawRightString(558, 28, str(doc.page))
    SimpleDocTemplate(str(pdf_path), pagesize=(612,792), leftMargin=54,
        rightMargin=54, topMargin=48, bottomMargin=58,
        title='Home Credit: credit-risk modeling', author='Ryan Wu').build(story, onFirstPage=footer, onLaterPages=footer)
    files = [p for p in source.iterdir() if p.is_file() and p.suffix in {'.json','.csv','.png','.joblib'}]
    manifest = {'generated_at':now.isoformat(), 'training_rerun':False,
        'source_directory':str(source.relative_to(ROOT)),
        'sources':{str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,
                     'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)},
        'code':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [ROOT/'src/train.py', ROOT/'src/monitoring.py',
                          ROOT/'src/artifacts.py', ROOT/'clean_report.py']},
        'references':[{'title':label,'url':url} for label,url in REFERENCES]}
    (artifact_path(destination, 'manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    links = [("Browser report", 'index.html'), ("PDF report", 'results.pdf'),
             ("Editable report", 'report.md'), ("Source manifest", 'manifest.json')]
    (artifact_path(destination, 'START_HERE.md')).write_text(
        '# Home Credit results\n\n' + ' | '.join(
            f'[{label}]({artifact_path(destination, name).name})' for label, name in links)
        + '\n\nFiles use the experiment date and a fixed reading-order number. '
        'Regenerating the presentation keeps the experiment date.\n\n'
        '## Files\n\n- `figures/`: numbered charts.\n- `data/`: numbered tables and metrics.\n'
        f'- Full predictions and fitted models are local-only in `outputs/{source.name}/`; they are excluded from Git.\n\n'
        'Regenerate: `python3 clean_report.py`.\n')

    print(f'Report: {artifact_path(destination, "index.html")}\nPDF: {pdf_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    source = args.source or latest_run(ROOT/'outputs')
    build(source, args.output or ROOT/'reports'/source.name)
