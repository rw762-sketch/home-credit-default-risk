"""Turn out-of-fold default probabilities into lending decisions and money.

Every economic assumption lives in one place, ``Economics``, and every modelling
assumption in ``ASSUMPTIONS``; each run writes both to assumptions.csv and the
report, with its source and the direction it biases results.

Per approved loan with credit amount C (AMT_CREDIT):
- repaid (TARGET = 0): earns ``margin * C``;
- payment difficulty (TARGET = 1): with probability ``loss_share`` it ends in a
  loss of ``LGD * ead_share * C``; otherwise the borrower recovers and repays,
  earning ``margin * C``.

Approving applicant i is worth it when its expected value is positive:

    p_i < margin / (loss_share * (margin + LGD * ead_share))

With loss_share = ead_share = 1 this is margin / (margin + LGD). The cutoff needs
no labels and does not depend on C. Reads saved development predictions only.
"""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.artifacts import TZ, artifact_path, new_run
from src.compare_bureau import sha256
from src.monitoring import threshold_sweep

CAPACITIES = (.05, .10, .20)


@dataclass(frozen=True)
class Economics:
    """Unit economics of one approved loan, as shares of its credit amount.

    None of these are observed in the Home Credit data; see ECONOMIC_SOURCES.
    """
    margin: float = .10
    lgd: float = .45
    loss_share: float = 1.
    ead_share: float = 1.

    def __post_init__(self):
        if not self.margin > 0:
            raise ValueError('Margin must be positive.')
        for name in ['lgd', 'loss_share', 'ead_share']:
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f'{name} must be in (0, 1].')

    @property
    def loss_rate(self):
        """Expected loss per unit of credit for a loan with payment difficulty."""
        return self.loss_share * self.lgd * self.ead_share

    @property
    def bad_value(self):
        """Expected value per unit of credit for a loan with payment difficulty."""
        return (1 - self.loss_share) * self.margin - self.loss_rate

    @property
    def cutoff(self):
        """Break-even PD: approve below it."""
        return self.margin / (self.loss_share * (self.margin + self.lgd * self.ead_share))

    def value(self, outcome_or_pd):
        """Value per unit of credit for a label (realized) or a PD (expected)."""
        q = np.asarray(outcome_or_pd, dtype=float)
        return (1 - q) * self.margin + q * self.bad_value


ECONOMIC_SOURCES = {
    'margin': ('Assumed lifetime net profit on a repaid loan, after funding and operating costs; '
               'Home Credit pricing is not in the data.',
               'Probably low for high-rate consumer lending; a higher margin raises the cutoff.'),
    'lgd': ('Basel foundation-IRB supervisory LGD for senior unsecured exposures (45%).',
            'Unsecured consumer LGDs are often higher (60-80%), which would lower the cutoff.'),
    'loss_share': ('Share of TARGET = 1 loans that end in a loss; 1 treats every payment '
                   'difficulty as a loss.',
                   'TARGET flags lateness, not write-off, so 1 overstates losses and lowers the cutoff.'),
    'ead_share': ('Share of the original credit still owed at default; 1 uses the full AMT_CREDIT.',
                  'Borrowers repay part of the loan first, so 1 overstates losses and lowers the cutoff.'),
}

ASSUMPTIONS = (
    ('PD', 'Out-of-fold probabilities are calibrated (decile gaps under 0.007 on development data).',
     'If calibration drifts, the cutoff applies to distorted probabilities; checked by expected_loss_by_band.csv.'),
    ('Exposure', 'AMT_CREDIT is the amount at risk for every loan.',
     'Fees, insurance and top-ups are ignored.'),
    ('Horizon', 'One period: no discounting, loan terms, or timing of losses.',
     'Long-term loans look more profitable than they are.'),
    ('Capital', 'No regulatory capital charge or funding-cost differences by risk.',
     'Risky approvals look cheaper than they are.'),
    ('Pricing', 'One margin for every applicant; risk only decides approve/decline.',
     'Real lenders price risk, approving some applicants this rule declines.'),
    ('Customer value', 'A declined good customer costs only this loan\'s margin.',
     'Ignores lifetime value, so false positives look cheaper than they are.'),
    ('Observability', 'Outcomes are known for every applicant, including those a policy would decline.',
     'Real lenders never see declined applicants\' outcomes (reject inference).'),
)


def assumptions_table(economics):
    rows = [{'assumption': name, 'value': getattr(economics, name), 'source': source, 'bias': bias}
            for name, (source, bias) in ECONOMIC_SOURCES.items()]
    rows += [{'assumption': name, 'value': '', 'source': source, 'bias': bias}
             for name, source, bias in ASSUMPTIONS]
    return pd.DataFrame(rows)


def policy_outcome(name, approve, y, p, ead, economics, threshold=np.nan):
    """Realized and model-expected results of approving the rows in ``approve``."""
    approve = np.asarray(approve, dtype=bool)
    bad = approve & (y == 1)
    profit = (economics.value(y[approve]) * ead[approve]).sum()
    expected = (economics.value(p[approve]) * ead[approve]).sum()
    approved = int(approve.sum())
    return {'policy': name, 'threshold': float(threshold), 'approved': approved,
            'approval_rate': approved / len(y),
            'bad_rate_approved': float(y[approve].mean()) if approved else np.nan,
            'defaults_caught': int(((~approve) & (y == 1)).sum()),
            'good_rejected': int(((~approve) & (y == 0)).sum()),
            'approved_credit': float(ead[approve].sum()),
            'realized_loss': float(economics.loss_rate * ead[bad].sum()),
            'expected_loss': float((economics.loss_rate * p[approve] * ead[approve]).sum()),
            'realized_profit': float(profit), 'expected_profit': float(expected)}


def top_k_reject(p, ids, fraction):
    """Reject the riskiest share; applicant ID breaks exact score ties."""
    order = np.lexsort((ids, -p))
    approve = np.ones(len(p), dtype=bool)
    approve[order[:round(len(p) * fraction)]] = False
    return approve


def compare_policies(y, p, ead, ids, economics, f2_cutoff):
    e, t = economics, economics.cutoff
    rows = [policy_outcome('approve all', np.ones(len(y), bool), y, p, ead, e),
            policy_outcome('approve if PD < 0.5', p < .5, y, p, ead, e, .5),
            policy_outcome('F2-selected cutoff', p < f2_cutoff, y, p, ead, e, f2_cutoff),
            *[policy_outcome(f'reject riskiest {c:.0%}', top_k_reject(p, ids, c), y, p, ead, e)
              for c in CAPACITIES],
            policy_outcome('cost-based cutoff', p < t, y, p, ead, e, t)]
    curve = profit_curve(y, p, ead, e, extra=(t, f2_cutoff, .5))
    best = curve.loc[curve.realized_profit.idxmax()]
    rows.append(policy_outcome('best observed cutoff (oracle)', p < best.threshold, y, p, ead,
                               e, best.threshold))
    return pd.DataFrame(rows), curve


def profit_curve(y, p, ead, economics, points=200, extra=()):
    """Realized profit when approving PD < t, for t on a grid of score quantiles.

    ``extra`` adds named policy cutoffs, so the best grid point bounds them too.
    """
    grid = np.unique(np.r_[np.quantile(p, np.linspace(0, 1, points)), list(extra), 1.000001])
    order = np.argsort(p, kind='stable')
    sorted_p = p[order]
    value = economics.value(y[order]) * ead[order]
    expected = economics.value(sorted_p) * ead[order]
    counts = np.searchsorted(sorted_p, grid, side='left')   # rows with PD < t
    cum = np.r_[0, np.cumsum(value)]
    cum_expected = np.r_[0, np.cumsum(expected)]
    return pd.DataFrame({'threshold': grid, 'approval_rate': counts / len(p),
                         'realized_profit': cum[counts], 'expected_profit': cum_expected[counts]})


def sensitivity(y, p, ead, margins, lgds, base=Economics()):
    """Vary margin and LGD; loss_share and ead_share stay at the base values."""
    rows = []
    for margin in margins:
        for lgd in lgds:
            e = Economics(margin, lgd, base.loss_share, base.ead_share)
            t = e.cutoff
            curve = profit_curve(y, p, ead, e, extra=(t,))
            best = curve.realized_profit.max()
            chosen = policy_outcome('cost-based', p < t, y, p, ead, e, t)
            everyone = policy_outcome('approve all', np.ones(len(y), bool), y, p, ead, e)
            rows.append({'margin': margin, 'lgd': lgd, 'loss_share': e.loss_share,
                         'ead_share': e.ead_share, 'cost_cutoff': t,
                         'approval_rate': chosen['approval_rate'],
                         'bad_rate_approved': chosen['bad_rate_approved'],
                         'profit_cost_based': chosen['realized_profit'],
                         'profit_approve_all': everyone['realized_profit'],
                         'profit_oracle': float(best),
                         'share_of_oracle': chosen['realized_profit'] / best if best > 0 else np.nan})
    return pd.DataFrame(rows)


def loss_by_band(y, p, ead, economics, bands=10):
    """Predicted vs realized loss by score decile: a calibration check in money."""
    frame = pd.DataFrame({'y': y, 'p': p, 'ead': ead})
    frame['band'] = pd.qcut(frame.p.rank(method='first'), bands, labels=range(1, bands+1))
    table = frame.groupby('band', observed=True).agg(
        applicants=('y', 'size'), mean_pd=('p', 'mean'), observed_rate=('y', 'mean'),
        mean_credit=('ead', 'mean'))
    rate = economics.loss_rate
    table['expected_loss'] = frame.assign(v=rate*frame.p*frame.ead).groupby('band', observed=True).v.sum()
    table['realized_loss'] = frame.assign(v=rate*frame.y*frame.ead).groupby('band', observed=True).v.sum()
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
    economics = Economics(args.margin, args.lgd, getattr(args, 'loss_share', 1.),
                          getattr(args, 'ead_share', 1.))
    predictions = args.predictions
    if predictions is None:
        predictions = artifact_path(args.source, 'development_predictions.csv')
    y, p, ead, ids = load(predictions, args.train, args.variant)
    _, f2 = threshold_sweep(y, p)
    policies, curve = compare_policies(y, p, ead, ids, economics, f2)
    grid = sensitivity(y, p, ead, args.margins, args.lgds, economics)
    bands = loss_by_band(y, p, ead, economics)
    assumptions = assumptions_table(economics)
    out = args.output or new_run('decision')
    if args.output:
        out.mkdir(parents=True, exist_ok=False)
    for name, table in [('policy_comparison.csv', policies), ('profit_curve.csv', curve),
                        ('sensitivity.csv', grid), ('expected_loss_by_band.csv', bands),
                        ('assumptions.csv', assumptions)]:
        table.to_csv(artifact_path(out, name), index=False)
    t = economics.cutoff
    chosen = policies.set_index('policy').loc['cost-based cutoff']
    everyone = policies.set_index('policy').loc['approve all']
    oracle = policies.set_index('policy').loc['best observed cutoff (oracle)']
    metrics = {'created_at': datetime.now(TZ).isoformat(), 'variant': args.variant,
               'applicants': int(len(y)), 'default_rate': float(y.mean()),
               'economics': asdict(economics),
               'assumptions': assumptions.to_dict(orient='records'),
               'cost_cutoff': t, 'f2_cutoff': float(f2),
               'cost_based': {k: (None if pd.isna(v) else v) for k, v in chosen.items()},
               'expected_vs_realized_loss_all': {'expected': float((economics.loss_rate*p*ead).sum()),
                                                 'realized': float((economics.loss_rate*y*ead).sum())},
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
    e = economics
    rule = (f'{e.margin}/({e.margin}+{e.lgd})' if e.loss_share == e.ead_share == 1 else
            f'{e.margin}/({e.loss_share}*({e.margin}+{e.lgd}*{e.ead_share}))')
    assumption_lines = ['| Assumption | Value | Source | Bias |', '| --- | ---: | --- | --- |',
                        *[f'| {r.assumption} | {r.value} | {r.source} | {r.bias} |'
                          for r in assumptions.itertuples()]]
    report = [
        '# From default probabilities to lending decisions', '',
        f'Variant `{args.variant}`: {len(y):,} development applicants, default rate {y.mean():.2%}. '
        f'Margin {e.margin:.0%} of credit on repaid loans; LGD {e.lgd:.0%} of the {e.ead_share:.0%} still '
        f'owed, for the {e.loss_share:.0%} of payment difficulties assumed to end in a loss. '
        'Amounts are in the dataset\'s currency units.', '',
        f'The cost-based rule approves when PD < {rule} = **{t:.4f}**. '
        f'It approves {chosen.approval_rate:.1%} of applicants with a {chosen.bad_rate_approved:.2%} bad rate, '
        f'versus {y.mean():.2%} when approving everyone. Realized profit changes from '
        f'{money(everyone.realized_profit)} to {money(chosen.realized_profit)}; the best cutoff found with '
        f'hindsight reaches {money(oracle.realized_profit)}.', '',
        *table, '',
        'The oracle row picks its cutoff using the outcomes it is scored on, so it is an upper bound, not a '
        'policy. The F2 cutoff and capacity rules ignore loan economics; the cost-based cutoff uses only the '
        'assumptions below.', '',
        'See sensitivity.csv for how the cutoff, approval rate and profit move across margin and LGD, and '
        'expected_loss_by_band.csv for predicted vs realized loss by score decile.', '',
        '## Assumptions', '',
        'None of the economic values are observed in the Home Credit data. Each row says where the value '
        'comes from and which way it biases the result.', '',
        *assumption_lines, '',
        'These are development OOF predictions that earlier experiments already inspected.', '']
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
    defaults = Economics()
    parser.add_argument('--margin', type=float, default=defaults.margin,
                        help='Lifetime net profit on a repaid loan, as a share of credit.')
    parser.add_argument('--lgd', type=float, default=defaults.lgd,
                        help='Share of the outstanding amount lost on a loss (Basel F-IRB: 0.45).')
    parser.add_argument('--loss-share', type=float, default=defaults.loss_share,
                        help='Share of payment difficulties (TARGET = 1) that end in a loss.')
    parser.add_argument('--ead-share', type=float, default=defaults.ead_share,
                        help='Share of the original credit still owed at default.')
    parser.add_argument('--margins', type=float, nargs='+', default=[.05, .10, .15, .20])
    parser.add_argument('--lgds', type=float, nargs='+', default=[.30, .45, .60, .75])
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
