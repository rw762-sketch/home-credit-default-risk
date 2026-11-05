"""Label-free aggregates of earlier Home Credit applications decided before day 0."""
from contextlib import contextmanager
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from src.bureau import _identifiers

SOURCE_NAME = 'previous_application.csv'
INPUT_COLUMNS = ('SK_ID_PREV', 'SK_ID_CURR', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION',
                 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_ANNUITY', 'CNT_PAYMENT')
KNOWN_STATUSES = ('Approved', 'Refused', 'Canceled', 'Unused offer')
FEATURE_COLUMNS = (
    'PREV_APPLICATION_COUNT', 'PREV_STATUS_KNOWN_COUNT', 'PREV_APPROVED_COUNT',
    'PREV_REFUSED_COUNT', 'PREV_CANCELED_COUNT', 'PREV_REFUSED_FRACTION',
    'PREV_APPLICATION_365_COUNT', 'PREV_REFUSED_365_COUNT',
    'PREV_DAYS_SINCE_LAST_DECISION', 'PREV_DAYS_SINCE_LAST_APPROVAL',
    'PREV_DAYS_SINCE_LAST_REFUSAL', 'PREV_APPROVED_CREDIT_SUM',
    'PREV_APPROVED_ANNUITY_SUM', 'PREV_APPROVED_RATIO_RECORD_COUNT',
    'PREV_APPROVED_CREDIT_APPLICATION_RATIO', 'PREV_APPROVED_CNT_PAYMENT_MEAN',
)
COUNT_COLUMNS = (
    'PREV_APPLICATION_COUNT', 'PREV_STATUS_KNOWN_COUNT', 'PREV_APPROVED_COUNT',
    'PREV_REFUSED_COUNT', 'PREV_CANCELED_COUNT', 'PREV_APPLICATION_365_COUNT',
    'PREV_REFUSED_365_COUNT', 'PREV_APPROVED_RATIO_RECORD_COUNT',
)
RECENT_DAYS = 365


@contextmanager
def _stream(path):
    path = Path(path)
    if path.suffix.lower() != '.zip':
        with path.open('rb') as stream:
            yield stream
        return
    with zipfile.ZipFile(path) as archive:
        members = [x for x in archive.infolist()
                   if not x.is_dir() and Path(x.filename).name == SOURCE_NAME]
        if len(members) != 1:
            raise ValueError(f'Archive must contain exactly one {SOURCE_NAME}.')
        with archive.open(members[0]) as stream:
            yield stream


def read_previous(path, applicant_ids, chunksize=250000):
    """Stream approved columns, retaining only the requested applicant IDs.

    No extraction, downloads, labels, or fitted state.
    """
    ids = _identifiers(pd.Series(applicant_ids), 'Application SK_ID_CURR', unique=True)
    parts, rows = [], 0
    with _stream(path) as stream:
        for chunk in pd.read_csv(stream, usecols=lambda c: c in INPUT_COLUMNS, chunksize=chunksize):
            missing = set(INPUT_COLUMNS) - set(chunk)
            if missing:
                raise ValueError(f'Missing previous-application columns: {sorted(missing)}')
            rows += len(chunk)
            parts.append(chunk.loc[chunk.SK_ID_CURR.isin(ids)].copy())
    result = (pd.concat(parts, ignore_index=True) if parts
              else pd.DataFrame(columns=list(INPUT_COLUMNS)))
    return result, {'source_rows': rows, 'scoped_rows': len(result), 'read_chunk_rows': chunksize}


def aggregate_previous(previous):
    """Return one row per applicant with eligible earlier applications, plus an audit.

    A record needs a known finite DAYS_DECISION <= 0. Statuses outside
    KNOWN_STATUSES count as applications with unknown status. Amounts that are
    negative or nonfinite are missing; the credit/application ratio uses only
    approved records with known credit and positive requested amount. Scheduled
    due/termination dates (and their 365243 sentinels) are not used.
    """
    missing = set(INPUT_COLUMNS) - set(previous.columns)
    if missing:
        raise ValueError(f'Missing previous-application columns: {sorted(missing)}')
    frame = previous.loc[:, list(INPUT_COLUMNS)].copy()
    day = pd.to_numeric(frame.DAYS_DECISION, errors='coerce').astype(float)
    reasons = {'unknown_or_nonfinite_decision_day': ~np.isfinite(day),
               'future_decision_day': np.isfinite(day) & day.gt(0)}
    excluded = pd.DataFrame(reasons, index=frame.index).any(axis=1)
    audit = {'scoped_rows': len(frame), 'excluded_rows': int(excluded.sum()),
             'exclusion_reasons_nonexclusive': {k: int(v.sum()) for k, v in reasons.items()},
             'time_boundary': 'application day 0; decision dates <= 0',
             'availability_limit': 'Supplied snapshot dates are screened; source provenance is not independently verified.',
             'unused_fields': 'Scheduled due/termination dates are not used.'}
    frame = frame.loc[~excluded].copy()
    # Ineligible future/unknown observations cannot affect key validation either.
    frame['SK_ID_CURR'] = _identifiers(frame.SK_ID_CURR, 'Previous SK_ID_CURR')
    _identifiers(frame.SK_ID_PREV, 'SK_ID_PREV', unique=True)
    frame['DAYS_DECISION'] = day.loc[~excluded]
    audit['amounts_masked'] = {}
    for col in ['AMT_APPLICATION', 'AMT_CREDIT', 'AMT_ANNUITY', 'CNT_PAYMENT']:
        values = pd.to_numeric(frame[col], errors='coerce').astype(float)
        invalid = ~np.isfinite(values) | values.lt(0)
        audit['amounts_masked'][col] = int((invalid & frame[col].notna()).sum())
        frame[col] = values.mask(invalid)
    status = frame.NAME_CONTRACT_STATUS.where(frame.NAME_CONTRACT_STATUS.isin(KNOWN_STATUSES))
    audit['unknown_status_rows'] = int(status.isna().sum())
    approved, refused = status.eq('Approved'), status.eq('Refused')
    recent = frame.DAYS_DECISION.ge(-RECENT_DAYS)
    paired = approved & frame.AMT_CREDIT.notna() & frame.AMT_APPLICATION.gt(0)
    frame = frame.assign(
        _known=status.notna().astype(float), _approved=approved.astype(float),
        _refused=refused.astype(float), _canceled=status.eq('Canceled').astype(float),
        _recent=recent.astype(float), _refused_recent=(refused & recent).astype(float),
        _approved_day=frame.DAYS_DECISION.where(approved),
        _refused_day=frame.DAYS_DECISION.where(refused),
        _approved_credit=frame.AMT_CREDIT.where(approved),
        _approved_annuity=frame.AMT_ANNUITY.where(approved),
        _approved_cnt=frame.CNT_PAYMENT.where(approved),
        _ratio_credit=frame.AMT_CREDIT.where(paired),
        _ratio_application=frame.AMT_APPLICATION.where(paired))
    grouped = frame.groupby('SK_ID_CURR', sort=True)
    result = grouped.agg(
        PREV_APPLICATION_COUNT=('SK_ID_PREV', 'size'),
        PREV_STATUS_KNOWN_COUNT=('_known', 'sum'),
        PREV_APPROVED_COUNT=('_approved', 'sum'),
        PREV_REFUSED_COUNT=('_refused', 'sum'),
        PREV_CANCELED_COUNT=('_canceled', 'sum'),
        PREV_APPLICATION_365_COUNT=('_recent', 'sum'),
        PREV_REFUSED_365_COUNT=('_refused_recent', 'sum'),
        PREV_DAYS_SINCE_LAST_DECISION=('DAYS_DECISION', 'max'),
        PREV_DAYS_SINCE_LAST_APPROVAL=('_approved_day', 'max'),
        PREV_DAYS_SINCE_LAST_REFUSAL=('_refused_day', 'max'),
        PREV_APPROVED_RATIO_RECORD_COUNT=('_ratio_credit', 'count'),
        PREV_APPROVED_CNT_PAYMENT_MEAN=('_approved_cnt', 'mean'))
    sums = grouped[['_approved_credit', '_approved_annuity', '_ratio_credit',
                    '_ratio_application']].sum(min_count=1)
    result['PREV_APPROVED_CREDIT_SUM'] = sums['_approved_credit']
    result['PREV_APPROVED_ANNUITY_SUM'] = sums['_approved_annuity']
    result['PREV_APPROVED_CREDIT_APPLICATION_RATIO'] = (
        sums['_ratio_credit'] / sums['_ratio_application'].where(lambda x: x > 0))
    result['PREV_REFUSED_FRACTION'] = (
        result.PREV_REFUSED_COUNT / result.PREV_STATUS_KNOWN_COUNT.where(lambda x: x > 0))
    for col in ['PREV_DAYS_SINCE_LAST_DECISION', 'PREV_DAYS_SINCE_LAST_APPROVAL',
                'PREV_DAYS_SINCE_LAST_REFUSAL']:
        result[col] = -result[col]
    result = result.loc[:, list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    audit['eligible_rows'] = len(frame)
    audit['applicants_with_eligible_history'] = len(result)
    return result.reset_index(), audit


def join_previous(applications, aggregates):
    """Left join without changing application count, order, index, or target."""
    ids = _identifiers(applications.SK_ID_CURR, 'Application SK_ID_CURR', unique=True)
    aggregate_ids = _identifiers(aggregates.SK_ID_CURR, 'Aggregate SK_ID_CURR', unique=True)
    if set(aggregates) != {'SK_ID_CURR', *FEATURE_COLUMNS}:
        raise ValueError('Previous-application aggregates must contain exactly the approved features and ID.')
    if set(FEATURE_COLUMNS) & set(applications):
        raise ValueError('Application columns collide with previous-application features.')
    extra = aggregates.assign(SK_ID_CURR=aggregate_ids).set_index('SK_ID_CURR').reindex(ids)
    extra.index = applications.index
    no_history = extra.PREV_APPLICATION_COUNT.isna()
    extra.loc[no_history, list(COUNT_COLUMNS)] = 0
    return pd.concat([applications.copy(), extra], axis=1)
