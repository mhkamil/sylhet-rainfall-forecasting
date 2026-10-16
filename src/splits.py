"""Stage 3 split protocol. Rows are feature dates t; every split is defined by the TARGET date t+1 (Decision 2.6).

Development = target dates 1981-2019. Locked test = target dates 2020-2025 (only via test_index(confirm_final=True)).
"""
import pandas as pd

DEV_LAST_TARGET_YEAR = 2019
TEST_YEARS = (2020, 2025)
# (last training target year, first validation year, last validation year)
CV_FOLDS = [(2004, 2005, 2007), (2007, 2008, 2010), (2010, 2011, 2013), (2013, 2014, 2016), (2016, 2017, 2019)]
DL_SPLIT = (2014, 2015, 2019)
GAP_DAYS = 1


def target_dates(index):
    return pd.DatetimeIndex(index) + pd.Timedelta(days=1)


def dev_index(index):
    """All rows whose target falls in 1981-2019."""
    return index[target_dates(index).year <= DEV_LAST_TARGET_YEAR]


def make_split(index, train_end_year, val_start_year, val_end_year, gap_days=GAP_DAYS):
    """Expanding-window split by target year; the last `gap_days` training rows are purged, so at least
    `gap_days` calendar days lie between the last training target and the first validation target."""
    if val_end_year > DEV_LAST_TARGET_YEAR:
        raise ValueError("validation years must lie in the development period (<= 2019)")
    ty = target_dates(index).year
    train = index[ty <= train_end_year]
    train = train[: len(train) - gap_days] if gap_days else train
    val = index[(ty >= val_start_year) & (ty <= val_end_year)]
    return train, val


def cv_folds(index, gap_days=GAP_DAYS):
    """The 5 expanding-window folds as a list of (train_index, val_index)."""
    return [make_split(index, *f, gap_days=gap_days) for f in CV_FOLDS]


def dl_split(index, gap_days=GAP_DAYS):
    """Single split for deep-learning models: train 1981-2014, validate 2015-2019."""
    return make_split(index, *DL_SPLIT, gap_days=gap_days)


def test_index(index, confirm_final=False):
    """Rows whose target falls in 2020-2025. Refuses to run unless the researcher has called final evaluation."""
    if confirm_final is not True:
        raise PermissionError("The test set is locked. Pass confirm_final=True only at the final evaluation (rule 4).")
    ty = target_dates(index).year
    return index[(ty >= TEST_YEARS[0]) & (ty <= TEST_YEARS[1])]
