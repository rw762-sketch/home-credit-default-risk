"""Turn out-of-fold default probabilities into lending decisions and money.

Stylized unit economics per approved loan with exposure EAD (AMT_CREDIT):
a loan that is repaid earns ``margin * EAD``; a loan with repayment
difficulty loses ``LGD * EAD``. Approving applicant i is worth it when

    (1 - p_i) * margin - p_i * LGD > 0   <=>   p_i < margin / (margin + LGD),

so the cost-based cutoff needs no labels and does not depend on EAD. Margin and
LGD are assumptions, not Home Credit figures, and TARGET flags payment
difficulty rather than a measured loss; the sensitivity grid shows how much the
conclusions depend on them. Reads saved development predictions only.
"""
import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import TZ, artifact_path, new_run
from src.compare_bureau import sha256
from src.monitoring import threshold_sweep

DEFAULT_LGD = .45      # Basel foundation-IRB LGD for senior unsecured exposures
DEFAULT_MARGIN = .10   # Assumed lifetime net margin on a repaid loan, as a share of credit
CAPACITIES = (.05, .10, .20)


def cost_cutoff(margin, lgd):
    if margin <= 0 or lgd <= 0:
        raise ValueError('Margin and LGD must be positive.')
    return margin / (margin + lgd)


def policy_outcome(name, approve, y, p, ead, margin, lgd, threshold=np.nan):
    """Realized and model-expected results of approving the rows in ``approve``."""
    approve = np.asarray(approve, dtype=bool)
    good, bad = approve & (y == 0), approve & (y == 1)
    profit = margin * ead[good].sum() - lgd * ead[bad].sum()
    expected = (margin * (1-p[approve]) * ead[approve]).sum() - (lgd * p[approve] * ead[approve]).sum()
    approved = int(approve.sum())
    return {'policy': name, 'threshold': float(threshold), 'approved': approved,
            'approval_rate': approved / len(y),
            'bad_rate_approved': float(y[approve].mean()) if approved else np.nan,
            'defaults_caught': int(((~approve) & (y == 1)).sum()),
            'good_rejected': int(((~approve) & (y == 0)).sum()),
            'approved_credit': float(ead[approve].sum()),
            'realized_loss': float(lgd * ead[bad].sum()),
            'expected_loss': float((lgd * p[approve] * ead[approve]).sum()),
            'realized_profit': float(profit), 'expected_profit': float(expected)}


def top_k_reject(p, ids, fraction):
    """Reject the riskiest share; applicant ID breaks exact score ties."""
    order = np.lexsort((ids, -p))
    approve = np.ones(len(p), dtype=bool)
    approve[order[:round(len(p) * fraction)]] = False
    return approve


def compare_policies(y, p, ead, ids, margin, lgd, f2_cutoff):
    t = cost_cutoff(margin, lgd)
    rows = [policy_outcome('approve all', np.ones(len(y), bool), y, p, ead, margin, lgd),
            policy_outcome('approve if PD < 0.5', p < .5, y, p, ead, margin, lgd, .5),
            policy_outcome('F2-selected cutoff', p < f2_cutoff, y, p, ead, margin, lgd, f2_cutoff),
            *[policy_outcome(f'reject riskiest {c:.0%}', top_k_reject(p, ids, c), y, p, ead, margin, lgd)
              for c in CAPACITIES],
            policy_outcome('cost-based cutoff', p < t, y, p, ead, margin, lgd, t)]
    curve = profit_curve(y, p, ead, margin, lgd, extra=(t, f2_cutoff, .5))
    best = curve.loc[curve.realized_profit.idxmax()]
    rows.append(policy_outcome('best observed cutoff (oracle)', p < best.threshold, y, p, ead,
                               margin, lgd, best.threshold))
    return pd.DataFrame(rows), curve


def profit_curve(y, p, ead, margin, lgd, points=200, extra=()):
    """Realized profit when approving PD < t, for t on a grid of score quantiles.

    ``extra`` adds named policy cutoffs, so the best grid point bounds them too.
    """
    grid = np.unique(np.r_[np.quantile(p, np.linspace(0, 1, points)), list(extra), 1.000001])
    order = np.argsort(p, kind='stable')
    sorted_p = p[order]
    value = np.where(y[order] == 1, -lgd, margin) * ead[order]
    expected = ((1-sorted_p) * margin - sorted_p * lgd) * ead[order]
    counts = np.searchsorted(sorted_p, grid, side='left')   # rows with PD < t
    cum = np.r_[0, np.cumsum(value)]
    cum_expected = np.r_[0, np.cumsum(expected)]
    return pd.DataFrame({'threshold': grid, 'approval_rate': counts / len(p),
                         'realized_profit': cum[counts], 'expected_profit': cum_expected[counts]})


def sensitivity(y, p, ead, margins, lgds):
    rows = []
    for margin in margins:
        for lgd in lgds:
            t = cost_cutoff(margin, lgd)
            curve = profit_curve(y, p, ead, margin, lgd, extra=(t,))
            best = curve.realized_profit.max()
            chosen = policy_outcome('cost-based', p < t, y, p, ead, margin, lgd, t)
            everyone = policy_outcome('approve all', np.ones(len(y), bool), y, p, ead, margin, lgd)
            rows.append({'margin': margin, 'lgd': lgd, 'cost_cutoff': t,
                         'approval_rate': chosen['approval_rate'],
                         'bad_rate_approved': chosen['bad_rate_approved'],
                         'profit_cost_based': chosen['realized_profit'],
                         'profit_approve_all': everyone['realized_profit'],
                         'profit_oracle': float(best),
                         'share_of_oracle': chosen['realized_profit'] / best if best > 0 else np.nan})
    return pd.DataFrame(rows)


def loss_by_band(y, p, ead, lgd, bands=10):
    """Predicted vs realized loss by score decile: a calibration check in money."""
    frame = pd.DataFrame({'y': y, 'p': p, 'ead': ead})
    frame['band'] = pd.qcut(frame.p.rank(method='first'), bands, labels=range(1, bands+1))
    table = frame.groupby('band', observed=True).agg(
        applicants=('y', 'size'), mean_pd=('p', 'mean'), observed_rate=('y', 'mean'),
        mean_credit=('ead', 'mean'))
    table['expected_loss'] = frame.assign(v=lgd*frame.p*frame.ead).groupby('band', observed=True).v.sum()
    table['realized_loss'] = frame.assign(v=lgd*frame.y*frame.ead).groupby('band', observed=True).v.sum()
    return table.reset_index()


def load(predictions, train, variant):
    """Saved OOF predictions joined to each applicant's credit amount."""
    frame = pd.read_csv(predictions)
    if variant not in frame:
        options = [c for c in frame if c not in {'SK_ID_CURR', 'TARGET', 'fold'}]
        raise ValueError(f'Variant {variant!r} not in predictions; choose from {options}.')
    credit = pd.read_csv(train, usecols=['SK_ID_CURR', 'AMT_CREDIT'])
    merged = frame[['SK_ID_CURR', 'TARGET', variant]].merge(credit, on='SK_ID_CURR', how='left',
                                                            validate='one_to_one')
    if merged.AMT_CREDIT.isna().any() or (merged.AMT_CREDIT <= 0).any():
        raise ValueError('Every applicant needs a positive AMT_CREDIT.')
    p = merged[variant].to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Probabilities must be finite and in [0, 1].')
    return (merged.TARGET.to_numpy(int), p, merged.AMT_CREDIT.to_numpy(float),
            merged.SK_ID_CURR.to_numpy())


def run(args):
    if not 0 < args.margin or not 0 < args.lgd <= 1:
        raise ValueError('Use a positive margin and an LGD in (0, 1].')
    predictions = args.predictions
    if predictions is None:
        predictions = artifact_path(args.source, 'development_predictions.csv')
    y, p, ead, ids = load(predictions, args.train, args.variant)
    _, f2 = threshold_sweep(y, p)
    policies, curve = compare_policies(y, p, ead, ids, args.margin, args.lgd, f2)
    grid = sensitivity(y, p, ead, args.margins, args.lgds)
    bands = loss_by_band(y, p, ead, args.lgd)
    out = args.output or new_run('decision')
    if args.output:
        out.mkdir(parents=True, exist_ok=False)
    for name, table in [('policy_comparison.csv', policies), ('profit_curve.csv', curve),
                        ('sensitivity.csv', grid), ('expected_loss_by_band.csv', bands)]:
        table.to_csv(artifact_path(out, name), index=False)
    t = cost_cutoff(args.margin, args.lgd)
    chosen = policies.set_index('policy').loc['cost-based cutoff']
    everyone = policies.set_index('policy').loc['approve all']
    oracle = policies.set_index('policy').loc['best observed cutoff (oracle)']
    metrics = {'created_at': datetime.now(TZ).isoformat(), 'variant': args.variant,
               'applicants': int(len(y)), 'default_rate': float(y.mean()),
               'assumptions': {'margin': args.margin, 'lgd': args.lgd, 'ead': 'AMT_CREDIT',
                               'label': 'TARGET (payment difficulty) treated as a loss event'},
               'cost_cutoff': t, 'f2_cutoff': float(f2),
               'cost_based': {k: (None if pd.isna(v) else v) for k, v in chosen.items()},
               'expected_vs_realized_loss_all': {'expected': float((args.lgd*p*ead).sum()),
                                                 'realized': float((args.lgd*y*ead).sum())},
               'evaluation': 'Previously inspected development OOF predictions; descriptive, not fresh validation.'}
    manifest = {'inputs': {str(Path(x).resolve()): sha256(x) for x in [predictions, args.train]},
                'code': {n: sha256(Path(__file__).with_name(n)) for n in ['decision.py', 'monitoring.py']},
                'versions': {n: version(n) for n in ['numpy', 'pandas', 'scikit-learn']},
                'command': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    for name, data in [('metrics.json', metrics), ('manifest.json', manifest)]:
        artifact_path(out, name).write_text(json.dumps(data, indent=2, allow_nan=False, default=float)+'\n')
    money = lambda v: f'{v/1e9:,.3f}B'
    table = ['| Policy | Cutoff | Approval rate | Bad rate (approved) | Realized profit | Expected profit |',
             '| --- | ---: | ---: | ---: | ---: | ---: |',
             *[f'| {r.policy} | {"" if np.isnan(r.threshold) else f"{r.threshold:.4f}"} | '
               f'{r.approval_rate:.1%} | {r.bad_rate_approved:.2%} | {money(r.realized_profit)} | '
               f'{money(r.expected_profit)} |' for r in policies.itertuples()]]
    report = [
        '# From default probabilities to lending decisions', '',
        f'Variant `{args.variant}`: {len(y):,} development applicants, default rate {y.mean():.2%}. '
        f'Assumed margin {args.margin:.0%} of credit on repaid loans and LGD {args.lgd:.0%} on loans with '
        f'repayment difficulty; exposure = AMT_CREDIT. Amounts are in the dataset\'s currency units.', '',
        f'The cost-based rule approves when PD < {args.margin}/({args.margin}+{args.lgd}) = **{t:.4f}**. '
        f'It approves {chosen.approval_rate:.1%} of applicants with a {chosen.bad_rate_approved:.2%} bad rate, '
        f'versus {y.mean():.2%} when approving everyone. Realized profit changes from '
        f'{money(everyone.realized_profit)} to {money(chosen.realized_profit)}; the best cutoff found with '
        f'hindsight reaches {money(oracle.realized_profit)}.', '',
        *table, '',
        'The oracle row picks its cutoff using the outcomes it is scored on, so it is an upper bound, not a '
        'policy. The F2 cutoff and capacity rules ignore loan economics; the cost-based cutoff uses only the '
        'assumptions above.', '',
        'See sensitivity.csv for how the cutoff, approval rate and profit move across margin and LGD '
        'assumptions, and expected_loss_by_band.csv for predicted vs realized loss by score decile.', '',
        '- Margin and LGD are assumptions; Home Credit pricing and recoveries are not in the data.',
        '- TARGET marks payment difficulty, which is not the same as a written-off loss.',
        '- Rejected applicants\' outcomes would be unobserved in practice (reject inference is not addressed).',
        '- These are development OOF predictions that earlier experiments already inspected.', '']
    artifact_path(out, 'report.md').write_text('\n'.join(report))
    print('\n'.join(report[2:5]), flush=True)
    print(f'Artifacts: {out.resolve()}', flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--source', type=Path, help='A compare_history output folder with development predictions.')
    source.add_argument('--predictions', type=Path, help='A development_predictions CSV.')
    parser.add_argument('--variant', default='lgbm_application_bureau_installments_ext_previous')
    parser.add_argument('--train', type=Path, default=Path('data/raw/application_train.csv'))
    parser.add_argument('--margin', type=float, default=DEFAULT_MARGIN)
    parser.add_argument('--lgd', type=float, default=DEFAULT_LGD)
    parser.add_argument('--margins', type=float, nargs='+', default=[.05, .10, .15, .20])
    parser.add_argument('--lgds', type=float, nargs='+', default=[.30, .45, .60, .75])
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
