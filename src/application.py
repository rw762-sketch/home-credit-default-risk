"""Extended row-local application features.

Every feature uses only the applicant's own row: no labels, other applicants,
fitted statistics, or history tables. Kept separate from ``src.train.features``
so the published application-only benchmark and its code hash stay unchanged.
"""
import numpy as np
import pandas as pd

EXT_SOURCES = ('EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3')
FEATURE_COLUMNS = (
    'EXT_SOURCE_MEAN', 'EXT_SOURCE_MIN', 'EXT_SOURCE_MAX', 'EXT_SOURCE_STD',
    'EXT_SOURCE_PRODUCT', 'EXT_SOURCE_KNOWN_COUNT', 'EMPLOYED_BIRTH_RATIO',
    'GOODS_CREDIT_RATIO', 'INCOME_PER_FAMILY_MEMBER',
)
EMPLOYMENT_SENTINEL = 365243


def _numeric(frame, column):
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors='coerce').astype(float)
    return values.where(np.isfinite(values))


def _ratio(numerator, denominator):
    """Zero, missing, or nonfinite denominators give a missing ratio."""
    result = numerator / denominator.where(denominator != 0)
    return result.where(np.isfinite(result))


def add_application_features(frame):
    """Return a copy of ``frame`` with ``FEATURE_COLUMNS`` appended.

    External-score summaries use the sources known for that applicant; the
    standard deviation needs at least two and the product needs all three.
    The employment sentinel 365243 is treated as missing. The input frame,
    its index, and its row order are unchanged.
    """
    collisions = set(FEATURE_COLUMNS) & set(frame.columns)
    if collisions:
        raise ValueError(f'Application columns collide with extended features: {sorted(collisions)}')
    result = frame.copy()
    scores = pd.concat({c: _numeric(frame, c) for c in EXT_SOURCES}, axis=1)
    known = scores.notna().sum(axis=1)
    result['EXT_SOURCE_MEAN'] = scores.mean(axis=1)
    result['EXT_SOURCE_MIN'] = scores.min(axis=1)
    result['EXT_SOURCE_MAX'] = scores.max(axis=1)
    result['EXT_SOURCE_STD'] = scores.std(axis=1, ddof=0).where(known >= 2)
    result['EXT_SOURCE_PRODUCT'] = scores.prod(axis=1, skipna=False)
    result['EXT_SOURCE_KNOWN_COUNT'] = known.astype(float)
    employed = _numeric(frame, 'DAYS_EMPLOYED').replace(EMPLOYMENT_SENTINEL, np.nan)
    result['EMPLOYED_BIRTH_RATIO'] = _ratio(employed, _numeric(frame, 'DAYS_BIRTH'))
    result['GOODS_CREDIT_RATIO'] = _ratio(_numeric(frame, 'AMT_GOODS_PRICE'),
                                          _numeric(frame, 'AMT_CREDIT'))
    result['INCOME_PER_FAMILY_MEMBER'] = _ratio(_numeric(frame, 'AMT_INCOME_TOTAL'),
                                                _numeric(frame, 'CNT_FAM_MEMBERS'))
    return result
