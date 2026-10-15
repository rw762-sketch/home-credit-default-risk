"""Label-free bureau features from records available at application time (day 0)."""
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    'SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT', 'DAYS_CREDIT_UPDATE',
    'CREDIT_ACTIVE', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
    'AMT_CREDIT_SUM_OVERDUE',
)
INPUT_COLUMNS = (*REQUIRED_COLUMNS, 'DAYS_ENDDATE_FACT')
FEATURE_COLUMNS = (
    'BUREAU_RECORD_COUNT', 'BUREAU_ACTIVE_COUNT', 'BUREAU_ACTIVE_FRACTION',
    'BUREAU_CREDIT_SUM', 'BUREAU_DEBT_SUM', 'BUREAU_OVERDUE_SUM',
    'BUREAU_CREDIT_KNOWN_COUNT', 'BUREAU_DEBT_KNOWN_COUNT',
    'BUREAU_OVERDUE_KNOWN_COUNT', 'BUREAU_RATIO_RECORD_COUNT',
    'BUREAU_DEBT_CREDIT_RATIO', 'BUREAU_DAYS_SINCE_LAST_CREDIT',
    'BUREAU_DAYS_SINCE_FIRST_CREDIT', 'BUREAU_DAYS_SINCE_LAST_UPDATE',
)
COUNT_COLUMNS = (
    'BUREAU_RECORD_COUNT', 'BUREAU_CREDIT_KNOWN_COUNT',
    'BUREAU_DEBT_KNOWN_COUNT', 'BUREAU_OVERDUE_KNOWN_COUNT',
    'BUREAU_RATIO_RECORD_COUNT',
)


def read_bureau(path):
    """Read only approved columns from a CSV or one unambiguous bureau.csv in ZIP.

    No archive extraction, downloads, labels, balance histories, or fitted state.
    """
    path = Path(path)
    options = {'usecols': lambda name: name in INPUT_COLUMNS}
    if path.suffix.lower() != '.zip':
        return pd.read_csv(path, **options)
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist()
                   if not item.is_dir() and Path(item.filename).name == 'bureau.csv']
        if len(members) != 1:
            raise ValueError('Archive must contain exactly one bureau.csv.')
        with archive.open(members[0]) as stream:
            return pd.read_csv(stream, **options)


def _identifiers(values, name, unique=False):
    numeric = pd.to_numeric(values, errors='coerce')
    valid = numeric.notna() & np.isfinite(numeric) & (numeric > 0) & (numeric % 1 == 0)
    if not valid.all():
        raise ValueError(f'{name} must be nonmissing positive integer IDs.')
    if unique and numeric.duplicated().any():
        raise ValueError(f'{name} must be unique; duplicate IDs would inflate history.')
    return numeric


def aggregate_bureau(bureau, applicant_ids=None):
    """Return one row per eligible applicant and an audit of the filtering rules.

    DAYS_CREDIT and DAYS_CREDIT_UPDATE must both be finite and <= 0.
    A supplied actual closure date must be missing or finite and <= 0.
    Future *scheduled* end dates are not observed events and are never used.
    These rules screen supplied snapshots; they cannot reconstruct prior states
    or independently verify the provider's application-time availability.

    Amount sums preserve all-missing groups as NaN. Debt is a signed net balance;
    negative credit limits/overdue amounts and all nonfinite amounts are missing.
    The debt/credit ratio uses only loans with known debt and positive credit.
    """
    missing = set(REQUIRED_COLUMNS) - set(bureau.columns)
    if missing:
        raise ValueError(f'Missing bureau columns: {sorted(missing)}')
    normalized_ids = _identifiers(bureau.SK_ID_CURR, 'Bureau SK_ID_CURR')
    _identifiers(bureau.SK_ID_BUREAU, 'SK_ID_BUREAU', unique=True)
    frame = bureau.loc[:, [c for c in INPUT_COLUMNS if c in bureau]].copy()
    frame['SK_ID_CURR'] = normalized_ids
    source_rows = len(frame)
    if applicant_ids is not None:
        scoped_ids = _identifiers(pd.Series(applicant_ids), 'Application SK_ID_CURR', unique=True)
        frame = frame[frame.SK_ID_CURR.isin(scoped_ids)].copy()

    credit_day = pd.to_numeric(frame.DAYS_CREDIT, errors='coerce').astype(float)
    update_day = pd.to_numeric(frame.DAYS_CREDIT_UPDATE, errors='coerce').astype(float)
    reasons = {
        'unknown_or_nonfinite_credit_day': ~np.isfinite(credit_day),
        'future_credit_day': credit_day > 0,
        'unknown_or_nonfinite_update_day': ~np.isfinite(update_day),
        'future_update_day': update_day > 0,
    }
    if 'DAYS_ENDDATE_FACT' in frame:
        actual_end = pd.to_numeric(frame.DAYS_ENDDATE_FACT, errors='coerce').astype(float)
        reasons['invalid_actual_end_day'] = (
            frame.DAYS_ENDDATE_FACT.notna() & ~np.isfinite(actual_end))
        reasons['future_actual_end_day'] = actual_end > 0
    excluded = pd.DataFrame(reasons, index=frame.index).any(axis=1)
    audit = {
        'source_rows': source_rows, 'scoped_rows': len(frame),
        'eligible_rows': int((~excluded).sum()), 'excluded_rows': int(excluded.sum()),
        'exclusion_reasons_nonexclusive': {k: int(v.sum()) for k, v in reasons.items()},
        'time_boundary': 'application day 0; observed credit/update/closure dates <= 0',
        'availability_limit': 'Supplied snapshot dates are screened; source provenance is not independently verified.',
        'negative_debt_policy': 'Retain finite signed net debt; do not silently clip credit balances.',
    }
    frame = frame.loc[~excluded].copy()
    frame['DAYS_CREDIT'] = credit_day.loc[~excluded]
    frame['DAYS_CREDIT_UPDATE'] = update_day.loc[~excluded]
    audit['amounts_masked'] = {}
    for col in ['AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_OVERDUE']:
        values = pd.to_numeric(frame[col], errors='coerce').astype(float)
        invalid = ~np.isfinite(values)
        if col != 'AMT_CREDIT_SUM_DEBT':
            invalid |= values < 0
        audit['amounts_masked'][col] = int((invalid & frame[col].notna()).sum())
        frame[col] = values.mask(invalid)

    known_status = frame.CREDIT_ACTIVE.isin(['Active', 'Closed', 'Sold', 'Bad debt'])
    frame['_active'] = (frame.CREDIT_ACTIVE == 'Active').astype(float).where(known_status)
    paired = frame.AMT_CREDIT_SUM.gt(0) & frame.AMT_CREDIT_SUM_DEBT.notna()
    frame['_ratio_credit'] = frame.AMT_CREDIT_SUM.where(paired)
    frame['_ratio_debt'] = frame.AMT_CREDIT_SUM_DEBT.where(paired)
    grouped = frame.groupby('SK_ID_CURR', sort=True)
    result = grouped.agg(
        BUREAU_RECORD_COUNT=('SK_ID_BUREAU', 'size'),
        BUREAU_ACTIVE_FRACTION=('_active', 'mean'),
        BUREAU_CREDIT_KNOWN_COUNT=('AMT_CREDIT_SUM', 'count'),
        BUREAU_DEBT_KNOWN_COUNT=('AMT_CREDIT_SUM_DEBT', 'count'),
        BUREAU_OVERDUE_KNOWN_COUNT=('AMT_CREDIT_SUM_OVERDUE', 'count'),
        BUREAU_RATIO_RECORD_COUNT=('_ratio_credit', 'count'),
        BUREAU_DAYS_SINCE_LAST_CREDIT=('DAYS_CREDIT', 'max'),
        BUREAU_DAYS_SINCE_FIRST_CREDIT=('DAYS_CREDIT', 'min'),
        BUREAU_DAYS_SINCE_LAST_UPDATE=('DAYS_CREDIT_UPDATE', 'max'),
    )
    sums = grouped[['_active', 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT',
                    'AMT_CREDIT_SUM_OVERDUE', '_ratio_credit', '_ratio_debt']].sum(min_count=1)
    result = result.join(sums.rename(columns={
        '_active': 'BUREAU_ACTIVE_COUNT', 'AMT_CREDIT_SUM': 'BUREAU_CREDIT_SUM',
        'AMT_CREDIT_SUM_DEBT': 'BUREAU_DEBT_SUM',
        'AMT_CREDIT_SUM_OVERDUE': 'BUREAU_OVERDUE_SUM'}))
    result['BUREAU_DEBT_CREDIT_RATIO'] = (
        result.pop('_ratio_debt') / result.pop('_ratio_credit').where(lambda x: x > 0))
    for col in FEATURE_COLUMNS:
        if 'DAYS_SINCE' in col:
            result[col] = -result[col]
    result = result.loc[:, list(FEATURE_COLUMNS)].replace([np.inf, -np.inf], np.nan)
    audit['applicants_with_eligible_history'] = len(result)
    return result.reset_index(), audit


def join_bureau(applications, aggregates):
    """Left join without changing application count, order, index, or target."""
    application_ids = _identifiers(applications.SK_ID_CURR, 'Application SK_ID_CURR', unique=True)
    aggregate_ids = _identifiers(aggregates.SK_ID_CURR, 'Aggregate SK_ID_CURR', unique=True)
    if set(aggregates.columns) != {'SK_ID_CURR', *FEATURE_COLUMNS}:
        raise ValueError('Aggregates must contain exactly the approved bureau features and ID.')
    if set(FEATURE_COLUMNS) & set(applications.columns):
        raise ValueError('Application columns collide with bureau features.')
    extra = aggregates.assign(SK_ID_CURR=aggregate_ids).set_index('SK_ID_CURR').reindex(application_ids)
    extra.index = applications.index
    no_history = extra.BUREAU_RECORD_COUNT.isna()
    extra.loc[no_history, list(COUNT_COLUMNS)] = 0
    extra.loc[no_history, 'BUREAU_ACTIVE_COUNT'] = 0
    return pd.concat([applications.copy(), extra], axis=1)
