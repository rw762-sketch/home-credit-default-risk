"""Per-applicant reason codes from LightGBM tree contributions (TreeSHAP).

For each applicant the model's log-odds score splits exactly into a baseline
plus one contribution per input column. Positive contributions raise the
predicted default risk. The top positive ones are the applicant's reasons,
the information an adverse-action notice has to state.

Contributions explain the model, not the world: they say which inputs moved
this score, not what caused repayment difficulty.
"""
import numpy as np
import pandas as pd

FAMILIES = (('BUREAU_', 'Credit bureau history'), ('INST_', 'Installment payment history'),
            ('PREV_', 'Previous Home Credit applications'), ('EXT_SOURCE', 'External credit scores'))


def family(feature):
    return next((label for prefix, label in FAMILIES if feature.startswith(prefix)), 'Application details')


def output_groups(preprocess):
    """Map every preprocessed column back to the original input column it came from.

    Numeric columns pass through one-to-one; a categorical column fans out into
    one indicator per kept category (rare categories share one 'infrequent' column).
    """
    groups = []
    for name, transformer, columns in preprocess.transformers_:
        if transformer == 'drop' or len(columns) == 0:
            continue
        if name == 'numeric':
            groups.extend(columns)
            continue
        encoder = transformer.named_steps['encode']
        infrequent = getattr(encoder, 'infrequent_categories_', [None]*len(columns))
        for column, categories, rare in zip(columns, encoder.categories_, infrequent):
            width = len(categories) - (len(rare) - 1 if rare is not None else 0)
            groups.extend([column]*width)
    if len(groups) != len(preprocess.get_feature_names_out()):
        raise ValueError('Could not map preprocessed columns back to input columns.')
    return groups


def contributions(model, x):
    """Return (per-input-column contributions in log-odds, baseline log-odds).

    Verifies additivity: baseline + row sum equals the model's raw score.
    """
    booster = model['model'].booster_
    transformed = model['preprocess'].transform(x)
    raw = booster.predict(transformed, pred_contrib=True)
    baseline, values = raw[:, -1], raw[:, :-1]
    score = booster.predict(transformed, raw_score=True)
    if not np.allclose(values.sum(axis=1) + baseline, score, rtol=0, atol=1e-6):
        raise ValueError('Tree contributions do not add up to the model score.')
    groups = pd.Index(output_groups(model['preprocess']))
    grouped = pd.DataFrame(values, columns=groups).T.groupby(level=0, sort=False).sum().T
    return grouped[list(x.columns)], baseline


def _display(value):
    if isinstance(value, (float, np.floating)):
        return 'missing' if np.isnan(value) else f'{value:.4g}'
    return 'missing' if pd.isna(value) else str(value)


def reason_codes(model, x, ids, top_k=3):
    """One row per applicant: probability and the top-k risk-raising inputs.

    An applicant with fewer than k positive contributions gets empty slots.
    """
    if top_k < 1:
        raise ValueError('top_k must be at least 1.')
    x = x.reset_index(drop=True)
    values, baseline = contributions(model, x)
    probability = model.predict_proba(x)[:, 1]
    matrix = values.to_numpy()
    order = np.argsort(-matrix, axis=1, kind='stable')[:, :top_k]
    names = np.asarray(values.columns)
    rows = []
    for i in range(len(x)):
        row = {'SK_ID_CURR': ids[i], 'probability': float(probability[i]),
               'baseline_log_odds': float(baseline[i]), 'score_log_odds': float(baseline[i] + matrix[i].sum())}
        for rank, j in enumerate(order[i], start=1):
            positive = matrix[i, j] > 0
            feature = names[j] if positive else ''
            row.update({f'reason_{rank}': feature,
                        f'reason_{rank}_family': family(feature) if positive else '',
                        f'reason_{rank}_value': _display(x.at[i, feature]) if positive else '',
                        f'reason_{rank}_log_odds': float(matrix[i, j]) if positive else np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def reason_summary(codes, top_k=3):
    """How often each input is an applicant's main reason or one of the top k."""
    long = pd.concat([codes[[f'reason_{r}', f'reason_{r}_log_odds']].set_axis(['feature', 'log_odds'], axis=1)
                      .assign(rank=r) for r in range(1, top_k+1)], ignore_index=True)
    long = long.loc[long.feature.ne('')]
    if long.empty:
        return pd.DataFrame(columns=['feature', 'family', 'top_reason', 'in_top_k',
                                     'share_of_applicants', 'mean_log_odds'])
    table = long.groupby('feature').agg(top_reason=('rank', lambda r: int((r == 1).sum())),
                                        in_top_k=('rank', 'size'), mean_log_odds=('log_odds', 'mean'))
    table['share_of_applicants'] = table.in_top_k / len(codes)
    table['family'] = [family(f) for f in table.index]
    return (table.reset_index().sort_values(['top_reason', 'in_top_k'], ascending=False)
            [['feature', 'family', 'top_reason', 'in_top_k', 'share_of_applicants', 'mean_log_odds']])
