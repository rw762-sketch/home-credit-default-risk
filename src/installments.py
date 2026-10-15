"""Conservative application-time aggregates of observed installment payments."""
from contextlib import contextmanager
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from src.bureau import _identifiers

KEYS = ['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER']
INPUT_COLUMNS = (*KEYS, 'NUM_INSTALMENT_VERSION', 'DAYS_INSTALMENT',
                 'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT')
FEATURE_COLUMNS = (
    'INST_EVENT_COUNT', 'INST_OBLIGATION_COUNT', 'INST_LOAN_COUNT',
    'INST_LATE_PAYMENT_FRACTION', 'INST_PAYMENT_DELAY_MEAN', 'INST_PAYMENT_DELAY_MAX',
    'INST_NUMERIC_OBLIGATION_COUNT', 'INST_SHORTFALL_COUNT', 'INST_SHORTFALL_FRACTION',
    'INST_PAIRED_DUE_SUM', 'INST_PAIRED_PAYMENT_SUM', 'INST_SHORTFALL_SUM',
    'INST_SHORTFALL_RATIO', 'INST_SHORTFALL_365_FRACTION',
    'INST_DAYS_SINCE_LAST_PAYMENT', 'INST_DAYS_SINCE_LAST_LATE_PAYMENT',
)
COUNT_COLUMNS = ['INST_EVENT_COUNT', 'INST_OBLIGATION_COUNT', 'INST_LOAN_COUNT',
                 'INST_NUMERIC_OBLIGATION_COUNT', 'INST_SHORTFALL_COUNT']


@contextmanager
def _stream(path):
    path = Path(path)
    if path.suffix.lower() != '.zip':
        with path.open('rb') as stream:
            yield stream
        return
    with zipfile.ZipFile(path) as archive:
        members = [x for x in archive.infolist()
                   if not x.is_dir() and Path(x.filename).name == 'installments_payments.csv']
        if len(members) != 1:
            raise ValueError('Archive must contain exactly one installments_payments.csv.')
        with archive.open(members[0]) as stream:
            yield stream


def read_installments(path, applicant_ids, chunksize=250000):
    """Stream approved columns, retaining only the requested development IDs."""
    ids = _identifiers(pd.Series(applicant_ids), 'Application SK_ID_CURR', unique=True)
    parts, rows = [], 0
    with _stream(path) as stream:
        for chunk in pd.read_csv(stream, usecols=lambda c: c in INPUT_COLUMNS, chunksize=chunksize):
            missing = set(INPUT_COLUMNS) - set(chunk)
            if missing:
                raise ValueError(f'Missing installment columns: {sorted(missing)}')
            rows += len(chunk)
            parts.append(chunk.loc[chunk.SK_ID_CURR.isin(ids)].copy())
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=INPUT_COLUMNS)
    return result, {'source_rows': rows, 'scoped_rows': len(result), 'read_chunk_rows': chunksize}


def aggregate_installments(payments):
    """Aggregate past payment events, then obligations, then applicants.

    Require known finite due/payment dates <= day 0. Future/unknown payment rows
    never supply amounts, schedule details, or version choices. Therefore this
    is *observed past-payment* history, not a census of all unpaid obligations.
    Multiple payment tranches share one scheduled amount per obligation. Exclude
    ambiguous versions, conflicting schedules, and exact duplicate event groups
    because this source has no transaction ID to disambiguate them.
    """
    missing = set(INPUT_COLUMNS) - set(payments)
    if missing:
        raise ValueError(f'Missing installment columns: {sorted(missing)}')
    frame = payments.loc[:, list(INPUT_COLUMNS)].copy()
    _identifiers(frame.SK_ID_CURR, 'Installment SK_ID_CURR')
    _identifiers(frame.SK_ID_PREV, 'Installment SK_ID_PREV')
    for col in INPUT_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors='coerce').astype(float)
    due, paid = frame.DAYS_INSTALMENT, frame.DAYS_ENTRY_PAYMENT
    reasons = {
        'unknown_or_nonfinite_due_date': ~np.isfinite(due),
        'future_due_date': due > 0,
        'unknown_or_nonfinite_payment_date': ~np.isfinite(paid),
        'future_payment_date': paid > 0,
        'invalid_installment_number': (~np.isfinite(frame.NUM_INSTALMENT_NUMBER) |
                                       frame.NUM_INSTALMENT_NUMBER.lt(0) |
                                       frame.NUM_INSTALMENT_NUMBER.mod(1).ne(0)),
        'invalid_version': ~np.isfinite(frame.NUM_INSTALMENT_VERSION) | frame.NUM_INSTALMENT_VERSION.lt(0),
    }
    excluded = pd.DataFrame(reasons).any(axis=1)
    audit = {'scoped_rows': len(frame), 'time_or_key_excluded_rows': int(excluded.sum()),
             'exclusion_reasons_nonexclusive': {k: int(v.sum()) for k, v in reasons.items()},
             'scope': 'Observed payment rows with both due/payment dates known and <= application day 0.',
             'delay_definition': 'Observed payment-tranche delay max(payment day - due day, 0); not loan cure time.',
             'shortfall_definition': 'One scheduled amount minus all known observed tranches by day 0, floored at zero.',
             'availability_limit': 'No inference of all unpaid obligations; supplied snapshot provenance is not independently verified.'}
    frame = frame.loc[~excluded].copy()
    # Make every ambiguity decision only after filtering post-application events.
    duplicates = frame.duplicated(list(INPUT_COLUMNS), keep=False)
    frame['_duplicate'] = duplicates
    grouped = frame.groupby(KEYS, sort=False)
    ambiguous = (grouped.NUM_INSTALMENT_VERSION.transform('nunique').gt(1) |
                 grouped.DAYS_INSTALMENT.transform('nunique').gt(1) |
                 grouped.AMT_INSTALMENT.transform('nunique').gt(1) |
                 grouped._duplicate.transform('any'))
    audit['ambiguous_schedule_or_duplicate_rows_excluded'] = int(ambiguous.sum())
    frame = frame.loc[~ambiguous].copy()
    audit['eligible_payment_rows'] = len(frame)
    audit['amounts_masked'] = {}
    for col in ['AMT_INSTALMENT', 'AMT_PAYMENT']:
        invalid = ~np.isfinite(frame[col]) | frame[col].lt(0)
        audit['amounts_masked'][col] = int((invalid & frame[col].notna()).sum())
        frame[col] = frame[col].mask(invalid)
    frame['_delay'] = (frame.DAYS_ENTRY_PAYMENT - frame.DAYS_INSTALMENT).clip(lower=0)
    frame['_late'] = frame._delay.gt(0).astype(float)
    frame['_late_day'] = frame.DAYS_ENTRY_PAYMENT.where(frame._late.eq(1))
    events = frame.groupby('SK_ID_CURR', sort=True).agg(
        INST_EVENT_COUNT=('SK_ID_PREV', 'size'),
        INST_LOAN_COUNT=('SK_ID_PREV', 'nunique'),
        INST_LATE_PAYMENT_FRACTION=('_late', 'mean'),
        INST_PAYMENT_DELAY_MEAN=('_delay', 'mean'),
        INST_PAYMENT_DELAY_MAX=('_delay', 'max'),
        INST_DAYS_SINCE_LAST_PAYMENT=('DAYS_ENTRY_PAYMENT', 'max'),
        INST_DAYS_SINCE_LAST_LATE_PAYMENT=('_late_day', 'max'))
    grouped = frame.groupby(KEYS, sort=False)
    obligations = grouped.agg(due=('DAYS_INSTALMENT', 'first'), amount=('AMT_INSTALMENT', 'first'),
                              events=('AMT_PAYMENT', 'size'), known_paid=('AMT_PAYMENT', 'count'),
                              known_due=('AMT_INSTALMENT', 'count'))
    obligations['paid'] = grouped.AMT_PAYMENT.sum(min_count=1)
    complete = obligations.events.eq(obligations.known_paid) & obligations.events.eq(obligations.known_due)
    obligations['numeric'] = complete.astype(int)
    obligations['shortfall'] = (obligations.amount - obligations.paid).clip(lower=0).where(complete)
    # Relative tolerance avoids marking floating-point rounding noise as arrears.
    tolerance = 1e-8 * obligations.amount.clip(lower=1)
    obligations['shortfall_flag'] = obligations.shortfall.gt(tolerance).astype(float).where(complete)
    obligations['recent_flag'] = obligations.shortfall_flag.where(obligations.due.ge(-365))
    obligations['paired_due'] = obligations.amount.where(complete)
    obligations['paired_paid'] = obligations.paid.where(complete)
    by_app = obligations.groupby(level='SK_ID_CURR', sort=True)
    summary = by_app.agg(INST_OBLIGATION_COUNT=('events', 'size'),
                         INST_NUMERIC_OBLIGATION_COUNT=('numeric', 'sum'),
                         INST_SHORTFALL_FRACTION=('shortfall_flag', 'mean'),
                         INST_SHORTFALL_365_FRACTION=('recent_flag', 'mean'))
    sums = by_app[['shortfall_flag', 'paired_due', 'paired_paid', 'shortfall']].sum(min_count=1)
    summary = summary.join(sums.rename(columns={
        'shortfall_flag': 'INST_SHORTFALL_COUNT', 'paired_due': 'INST_PAIRED_DUE_SUM',
        'paired_paid': 'INST_PAIRED_PAYMENT_SUM', 'shortfall': 'INST_SHORTFALL_SUM'}))
    result = events.join(summary)
    result['INST_SHORTFALL_RATIO'] = result.INST_SHORTFALL_SUM / result.INST_PAIRED_DUE_SUM.where(lambda x: x > 0)
    for col in ['INST_DAYS_SINCE_LAST_PAYMENT', 'INST_DAYS_SINCE_LAST_LATE_PAYMENT']:
        result[col] = -result[col]
    result = result.loc[:, list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    audit['eligible_obligations'] = len(obligations)
    audit['applicants_with_eligible_history'] = len(result)
    return result.reset_index(), audit


def join_installments(applications, aggregates):
    ids = _identifiers(applications.SK_ID_CURR, 'Application SK_ID_CURR', unique=True)
    aggregate_ids = _identifiers(aggregates.SK_ID_CURR, 'Aggregate SK_ID_CURR', unique=True)
    if set(aggregates) != {'SK_ID_CURR', *FEATURE_COLUMNS}:
        raise ValueError('Installment aggregates must contain exactly the approved features and ID.')
    if set(FEATURE_COLUMNS) & set(applications):
        raise ValueError('Application columns collide with installment features.')
    extra = aggregates.assign(SK_ID_CURR=aggregate_ids).set_index('SK_ID_CURR').reindex(ids)
    extra.index = applications.index
    no_history = extra.INST_EVENT_COUNT.isna()
    extra.loc[no_history, COUNT_COLUMNS] = 0
    return pd.concat([applications.copy(), extra], axis=1)
