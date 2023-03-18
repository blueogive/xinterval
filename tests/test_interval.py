#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np
import pandas as pd
# import piso

# piso.register_accessors()

from timeit import default_timer

from extended_interval import XInterval
import datetime
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class FakeData:
    dimension: str
    series_shape: int
    anchor_date: pd.Timestamp
    start_date_offset: Tuple[int]
    end_date_offset: Tuple[int]
    series_name: str = "Id"
    unit: str = "days"


def make_intervals(
    conf: FakeData, id_array: np.ndarray, rs: np.random._generator.Generator
):
    dim_start_dt = [
        conf.anchor_date + pd.to_timedelta(offset, unit=conf.unit)
        for offset in rs.integers(
            conf.start_date_offset[0], conf.start_date_offset[1], conf.series_shape
        )
    ]
    dim_end_dt = [
        start_dt
        + pd.to_timedelta(
            rs.integers(conf.end_date_offset[0], conf.end_date_offset[1], 1),
            unit=conf.unit,
        )
        for start_dt in dim_start_dt
    ]
    # The line below performs a necessary, regretable type conversion.
    dim_end_dt = [pd.Timestamp(itm.date[0]) for itm in dim_end_dt]
    df = pd.DataFrame(
        {
            f"{conf.dimension}StartDate": dim_start_dt,
            f"{conf.dimension}EndDate": dim_end_dt,
            f"{conf.dimension}Interval": pd.IntervalIndex.from_tuples(
                [itm for itm in zip(dim_start_dt, dim_end_dt)], closed="left"
            ),
        },
        index=[
            id_array[offid]
            for offid in rs.integers(0, id_array.shape[0], conf.series_shape)
        ],
    )
    df.index.name = conf.series_name
    return df


contsup_conf = FakeData(
    dimension="Supervision",
    series_shape=750,
    anchor_date=pd.Timestamp("2019-01-01"),
    start_date_offset=(-720, 720),
    end_date_offset=(20, 720),
)

level_conf = FakeData(
    dimension="Level",
    series_shape=1000,
    anchor_date=pd.Timestamp("2019-01-01"),
    start_date_offset=(-720, 720),
    end_date_offset=(0, 90),
)

rs_ids: np.random._generator.Generator = np.random.default_rng(11010001)
ids = rs_ids.integers(1001, 240000, 10000)


@pytest.fixture
def supdf(
    config: FakeData = contsup_conf,
    ids: np.ndarray = ids,
    rs_intval: np.random._generator.Generator = np.random.default_rng(8291),
) -> pd.DataFrame:
    return make_intervals(config, ids, rs_intval)


@pytest.fixture
def lvldf(
    config: FakeData = level_conf,
    ids: np.ndarray = ids,
    rs_intval: np.random._generator.Generator = np.random.default_rng(923462),
) -> pd.DataFrame:
    return make_intervals(config, ids, rs_intval)


@pytest.fixture
def basedf():
    rng = np.random.default_rng(1)
    cases = 10
    idrange = cases // 2
    dtrange = -50
    offsets = np.array(
        [
            -321,
            -132,
            -109,
            -415,
            -101,
            -109,
            -50,
            -308,
            -203,
            -110,
            -310,
            -12,
            -10,
            -401,
            -98,
            -19,
            -45,
            -38,
            -23,
            -95,
        ],
        np.int32,
    )
    offsets = offsets.reshape(2, 10).T
    IDXCOLS = ["ID", "Start_Date", "End_Date"]
    # Construct example dataframe
    df1 = pd.DataFrame(
        {
            "ID": rng.integers(1, idrange, cases),
            "Start_Date": [
                datetime.date(year=2016, month=1, day=1)
                + datetime.timedelta(days=int(offset))
                for offset in offsets[:, 0]
            ],
            "End_Date": [
                datetime.date(year=2016, month=1, day=1)
                + datetime.timedelta(days=int(offset))
                for offset in offsets[:, 1]
            ],
        }
    )
    return df1


def pdintvaldf(cases: int, seed: int, offsets: Tuple[int]):
    rng = np.random.default_rng(seed)
    idrange = cases // 2
    ids = rng.integers(1, idrange, cases)
    offsets = rng.integers(offsets[0], offsets[1], cases * 2)
    offsets = offsets.reshape(2, cases).T
    IDXCOLS = ["Id", "StartDate", "EndDate"]
    # Construct example dataframe
    df1 = pd.DataFrame(
        {
            IDXCOLS[0]: ids,
            IDXCOLS[1]: [
                pd.Timestamp("2021-01-01") + pd.Timedelta(days=offset)
                for offset in offsets.min(axis=1)
            ],
            IDXCOLS[2]: [
                pd.Timestamp("2021-01-01") + pd.Timedelta(days=offset)
                for offset in offsets.max(axis=1)
            ],
        }
    )
    df1["Interval"] = pd.arrays.IntervalArray.from_arrays(
        df1[IDXCOLS[1]], df1[IDXCOLS[2]], closed="left"
    )
    df1 = df1.set_index("Interval")
    return df1.drop(columns=IDXCOLS[1:])


@pytest.fixture
def sup2df(cases: int = 10, seed: int = 11010001, offsets: Tuple[int] = (-100, 100)):
    return pdintvaldf(cases, seed, offsets)


@pytest.fixture
def lvl2df(cases: int = 10, seed: int = 11010001, offsets: Tuple[int] = (-10, 10)):
    return pdintvaldf(cases, seed, offsets)


@pytest.fixture
def intervaldf(basedf):
    basedf["Intval"] = basedf.apply(
        lambda x: pyinter.closedopen(x["Start_Date"], x["End_Date"]), axis=1
    )
    return basedf


@pytest.fixture
def intervalsetdf(intervaldf):
    df2 = intervaldf.groupby("ID").apply(lambda x: pyinter.IntervalSet(x["Intval"]))
    return df2.reset_index().rename(columns={0: "Intset"})
    # df4 = pyinter.interval2record(df3, idcol='ID', intvalcol='Intset')
    # pd.to_pickle(df4, 'interval2record0.pkl')


# def test_interval2record(intervalsetdf, id='ID'):
#     """Test the interval2record()."""
#     t0 = pyinter.interval2record(intervalsetdf, idcol=id, intvalcol='Intset')
#     # t0.to_pickle("tests/interval2record0.pkl")
#     t0tst = pd.read_pickle('tests/interval2record0.pkl')
#     cpcols = ['ID', 'Begin_Date', 'End_Date']
#     t0.sort_values(cpcols, inplace=True)
#     t0tst.sort_values(cpcols, inplace=True)
#     # assert t0[cpcols].equals(t0tst[cpcols])
#     assert np.all(t0[cpcols].values == t0tst[cpcols].values)


# def test_normalizespells_stock(intervaldf):
#     """Test the normalizespells() with stock cohort."""
#     testdf = pyinter.normalizespells(intervaldf, dimname='In_Cohort',
#                                      idcol='ID', intvalcol='Intval',
#                                      startdtcol='Start_Date',
#                                      enddtcol='End_Date', intvaltype='[)',
#                                      cohorttype='stock',
#                                      cohortintval=pyinter.closedopen(
#                                          datetime.date(2015, 2, 1),
#                                          datetime.date(2015, 9, 1)
#                                         )
#                                     )
#     # pd.to_pickle(testdf, 'tests/normalizespells0.pkl')
#     t10tst = pd.read_pickle('tests/normalizespells0.pkl')
#     cpcols = ['ID', 'Start_Date', 'End_Date', 'dimension']
#     testdf.sort_values(cpcols, inplace=True)
#     t10tst.sort_values(cpcols, inplace=True)
#     assert np.all(testdf[cpcols].values == t10tst[cpcols].values)


# def test_normalizespells_entry(basedf):
#     """Test the normalizespells() with entry cohort."""
#     testdf = pyinter.normalizespells(basedf, dimname='In_Cohort',
#                                      idcol='ID', intvalcol='Intval',
#                                      startdtcol='Start_Date',
#                                      enddtcol='End_Date', intvaltype='[)',
#                                      cohorttype='entry',
#                                      cohortintval=pyinter.closedopen(
#                                          datetime.date(2015, 2, 1),
#                                          datetime.date(2015, 9, 1)
#                                         )
#                                     )
#     # pd.to_pickle(testdf, 'tests/normalizespells1.pkl')
#     t10tst = pd.read_pickle('tests/normalizespells1.pkl')
#     cpcols = ['ID', 'Start_Date', 'End_Date', 'dimension']
#     testdf.sort_values(cpcols, inplace=True)
#     t10tst.sort_values(cpcols, inplace=True)
#     assert np.all(testdf[cpcols].values == t10tst[cpcols].values)


# def test_normalizespells_exit(basedf):
#     """Test the normalizespells() with exit cohort."""
#     testdf = pyinter.normalizespells(basedf, dimname='In_Cohort',
#                                      idcol='ID', intvalcol='Intval',
#                                      startdtcol='Start_Date',
#                                      enddtcol='End_Date', intvaltype='[)',
#                                      cohorttype='exit',
#                                      cohortintval=pyinter.closedopen(
#                                          datetime.date(2014, 11, 1),
#                                          datetime.date(2015, 3, 1)
#                                         )
#                                     )
#     # pd.to_pickle(testdf, 'tests/normalizespells2.pkl')
#     t10tst = pd.read_pickle('tests/normalizespells2.pkl')
#     cpcols = ['ID', 'Start_Date', 'End_Date', 'dimension']
#     testdf.sort_values(cpcols, inplace=True)
#     t10tst.sort_values(cpcols, inplace=True)
#     assert np.all(testdf[cpcols].values == t10tst[cpcols].values)


# def test_intervalidxoverlap_daily():
#     date = pd.date_range('20150101', '20150110', freq='D',
#                          name='Date', closed='left')
#     intval = pyinter.closedopen(datetime.datetime(2014, 12, 30),
#                                 datetime.datetime(2015, 1, 3))
#     rslt = pyinter.intervalidxoverlap(date, intval)
#     assert rslt == (0, 2)


# def test_intervalidxoverlap_weekly_closed_open():
#     date = pd.date_range('20210401', '20210509', freq='W',
#                          name='Date', closed='left')
#     intval = pyinter.closedopen(datetime.datetime(2021, 3, 10),
#                                 datetime.datetime(2021, 5, 9))
#     rslt = pyinter.intervalidxoverlap(date, intval)
#     assert rslt == (0, 5)


# def test_intervalidxoverlap_weekly_closed():
#     date = pd.date_range('20210401', '20210517', freq='W',
#                          name='Date', closed='left')
#     intval = pyinter.closed(datetime.datetime(2021, 3, 10),
#                             datetime.datetime(2021, 5, 8))
#     rslt = pyinter.intervalidxoverlap(date, intval)
#     assert rslt == (0, 5)


def test_intervalagg(basedf):
    dtrng = pd.date_range("2015-01-01", "2016-01-01", freq="D")
    intvls = [
        pd.Interval(val[0], val[1], closed="left") for val in zip(dtrng[:-1], dtrng[1:])
    ]
    rowidx = pd.IntervalIndex(intvls, closed="left")
    colidx = basedf["ID"].unique()
    ary = np.zeros((len(intvls), len(colidx)), dtype=np.int8, order="C")
    df = pd.DataFrame(ary, index=rowidx, columns=colidx)
    sdf = df.astype(pd.SparseDtype(np.int8, 0))
    # sdf = df.astype(pd.SparseDtype(np.int16, 0))
    for row in basedf.iterrows():
        origdtype = sdf[row[1]["ID"]].dtypes
        sdf[row[1]["ID"]] = sdf[row[1]["ID"]].sparse.to_dense()
        sdf.loc[
            sdf.index.overlaps(
                pd.Interval(
                    pd.Timestamp(row[1]["Start_Date"]),
                    pd.Timestamp(row[1]["End_Date"]),
                    closed="left",
                )
            ),
            row[1]["ID"],
        ] += 1
        sdf[row[1]["ID"]] = sdf[row[1]["ID"]].astype(origdtype)
    sdf2 = sdf.where(
        sdf <= 1, 1
    )  # the pandas interval class does not remove overlapping intervals; this line adjusts for that by top-coding the matrix values to 1.
    # breakpoint()
    assert np.all(sdf2.sort_index(axis=1).sum().values == np.array([184, 26, 120, 298]))
    assert np.all(
        sdf2.loc[
            "2015-02-25":"2015-03-01",
        ]
        .sum(axis=1)
        .values
        == np.array([0, 0, 1, 1, 1])
    )


def _closed_override(row: pd.Series, intervalcol: str) -> bool:
    if row[intervalcol].is_empty:
        if row[intervalcol].closed_left or row[intervalcol].closed_right:
            return True
    return False


def _check_interval_order(
    row: pd.Series, intervalcol_left: str, intervalcol_right: str
) -> Tuple[str]:
    """Ensure the intervalcol_left is to the left of (on the left-to-right number line or time line) the intervalcol_right. If not, reverse the column names."""
    if (
        (row[intervalcol_right].left < row[intervalcol_left].left)
        or (
            row[intervalcol_right].left == row[intervalcol_left].left
            and row[intervalcol_right].closed_left
            and row[intervalcol_left].open_left
        )
        or (
            row[intervalcol_right].left == row[intervalcol_left].left
            and row[intervalcol_right].length > row[intervalcol_left].length
        )
    ):
        return (intervalcol_right, intervalcol_left)
    return (intervalcol_left, intervalcol_right)


def overlaps(row: pd.Series, intervalcol_left: str, intervalcol_right: str) -> bool:
    # if _closed_override(row, intervalcol_left) and _closed_override(row, intervalcol_right):
    #     return False
    # if _closed_override(row, intervalcol_left):
    #     if row[intervalcol_left].left == row[intervalcol_right].left and row[intervalcol_right].closed_left:
    #         return True
    #     elif row[intervalcol_left].left == row[intervalcol_right].right and row[intervalcol_right].closed_right:
    #         return True
    # elif _closed_override(row, intervalcol_right):
    #     if row[intervalcol_right].left == row[intervalcol_left].left and row[intervalcol_left].closed_left:
    #         return True
    #     elif row[intervalcol_right].left == row[intervalcol_left].right and row[intervalcol_left].closed_right:
    #         return True
    return row[intervalcol_left].overlaps(row[intervalcol_right])


def adjacent(row: pd.Series, intervalcol_left: str, intervalcol_right: str) -> bool:
    intervalcol_left, intervalcol_right = _check_interval_order(
        row, intervalcol_left, intervalcol_right
    )
    if row[intervalcol_left].overlaps(row[intervalcol_right]):
        return False
    elif row[intervalcol_left].is_empty and (
        row[intervalcol_left].left == row[intervalcol_right].left
        or row[intervalcol_left].left == row[intervalcol_right].right
    ):
        return True
    elif row[intervalcol_right].is_empty and (
        row[intervalcol_right].left == row[intervalcol_left].left
        or row[intervalcol_right].left == row[intervalcol_left].right
    ):
        return True
    elif row[intervalcol_left].right < row[intervalcol_right].left:
        if row[intervalcol_left].open_right ^ row[intervalcol_right].open_left:
            if pd.Interval(
                row[intervalcol_left].left, row[intervalcol_left].right, closed="right"
            ).overlaps(
                pd.Interval(
                    row[intervalcol_right].left,
                    row[intervalcol_right].right,
                    closed="left",
                )
            ):
                return True
    return False


def intersection_piso(
    row: pd.Series, intervalcol_left: str, intervalcol_right: str
) -> pd.IntervalDtype:
    try:
        return piso.intersection(
            row[intervalcol_left], row[intervalcol_right], squeeze=False
        )
    except TypeError:
        return pd.NA


def union_piso(
    row: pd.Series, intervalcol_left: str, intervalcol_right: str
) -> pd.IntervalDtype:
    try:
        return piso.union(row[intervalcol_left], row[intervalcol_right], squeeze=False)
    except TypeError:
        return pd.NA


def intersection(
    row: pd.Series, intervalcol_left: str, intervalcol_right: str
) -> pd.IntervalDtype:
    if row[intervalcol_left].overlaps(row[intervalcol_right]):
        leftval = (
            row[intervalcol_right].left
            if row[intervalcol_left].left < row[intervalcol_right].left
            else row[intervalcol_right].left
        )
        leftopen = False
        leftopen = (
            row[intervalcol_left].open_left and leftval == row[intervalcol_left]
        ) or (row[intervalcol_right].open_left and leftval == row[intervalcol_right])
        rightval = (
            row[intervalcol_left].right
            if row[intervalcol_left].right < row[intervalcol_right].right
            else row[intervalcol_right].right
        )
        rightopen = False
        rightopen = (
            row[intervalcol_left].open_right and rightval == row[intervalcol_left].right
        ) or (
            row[intervalcol_right].open_right
            and rightval == row[intervalcol_right].right
        )
        closedval = "right"
        if leftopen and rightopen:
            closedval = "neither"
        elif not leftopen and rightopen:
            closedval = "left"
        elif not leftopen and not rightopen:
            closedval = "both"
        return pd.Interval(leftval, rightval, closedval)
    else:
        return pd.Interval(
            row[intervalcol_left].left, row[intervalcol_left].left, "neither"
        )


def union(
    row: pd.Series, intervalcol_left: str, intervalcol_right: str
) -> pd.IntervalDtype:
    intervalcol_left, intervalcol_right = _check_interval_order(
        row, intervalcol_left, intervalcol_right
    )
    if row[intervalcol_left].overlaps(row[intervalcol_right]) or adjacent(
        row, intervalcol_left, intervalcol_right
    ):
        rightval = (
            row[intervalcol_left].right
            if row[intervalcol_left].right > row[intervalcol_right].right
            else row[intervalcol_right].right
        )
        leftopen = False
        leftopen = row[intervalcol_left].open_left or (
            row[intervalcol_left].left == row[intervalcol_right].left
            and row[intervalcol_right].open_left
        )
        rightopen = False
        rightopen = row[intervalcol_left].open_right or (
            row[intervalcol_left].right == row[intervalcol_right].right
            and row[intervalcol_right].open_right
        )
        closedval = "right"
        if leftopen and rightopen:
            closedval = "neither"
        elif not leftopen and rightopen:
            closedval = "left"
        elif not leftopen and not rightopen:
            closedval = "both"
        return pd.Interval(row[intervalcol_left].left, rightval, closed=closedval)
    else:
        # closedval = "right"
        # if row[intervalcol_left] and rightopen:
        #     closedval = "neither"
        # elif not leftopen and rightopen:
        #     closedval = "left"
        # elif not leftopen and not rightopen:
        #     closedval = "both"
        return pd.arrays.IntervalArray(
            (row[intervalcol_left], row[intervalcol_right]),
            closed=row[intervalcol_left].closed,
        )


def test_pdinterval_ops(sup2df, lvl2df):
    supdf = sup2df.reset_index().set_index("Id").rename(columns={"Interval": "SupervisionInterval"})
    lvldf = lvl2df.reset_index().set_index("Id").rename(columns={"Interval": "LevelInterval"})
    df = pd.merge(supdf, lvldf, how="inner", left_index=True, right_index=True)
    durations = []
    durations.append(default_timer())
    df["_overlaps"] = df.apply(
        overlaps, axis=1, args=("SupervisionInterval", "LevelInterval")
    )
    durations.append(default_timer())
    df["_adjacent"] = df.apply(
        adjacent, axis=1, args=("SupervisionInterval", "LevelInterval")
    )
    durations.append(default_timer())
    df["_intersection"] = df.apply(
        intersection, axis=1, args=("SupervisionInterval", "LevelInterval")
    )
    durations.append(default_timer())
    df["_union"] = df.apply(
        union, axis=1, args=("SupervisionInterval", "LevelInterval")
    )
    durations.append(default_timer())
    durarray = np.diff(np.array(durations), n=1)
    methodlst = ["overlaps", "adjacent", "intersection", "union"]
    durationsdct = {m: v for m, v in zip(methodlst, durarray)}
    collst = [
        "SupervisionInterval",
        "LevelInterval",
        "_overlaps",
        "_adjacent",
        "_intersection",
        "_union",
    ]
    # df[collst].head(10)
    # df.loc[df["_adjacent"], collst].head(10)
    breakpoint()
    df.loc[~df["_intersection"].array.is_empty]
    return None


def test_xinterval(supdf, lvldf):
    xint = XInterval()
    sdf = supdf.reset_index()
    ldf = lvldf.reset_index()
    durations = []
    durations.append(default_timer())
    df = xint.overlaps(left=sdf, right=ldf, how="left", left_on="Id", right_on="Id")
    durations.append(default_timer())
    # df["_adjacent"] = df.apply(
    #     adjacent, axis=1, args=("SupervisionInterval", "LevelInterval")
    # )
    # durations.append(default_timer())
    # df["_intersection"] = df.apply(
    #     intersection, axis=1, args=("SupervisionInterval", "LevelInterval")
    # )
    # durations.append(default_timer())
    # df["_union"] = df.apply(
    #     union, axis=1, args=("SupervisionInterval", "LevelInterval")
    # )
    # durations.append(default_timer())
    durarray = np.diff(np.array(durations), n=1)
    methodlst = ["overlaps", "adjacent", "intersection", "union"]
    durationsdct = {m: v for m, v in zip(methodlst, durarray)}
    collst = [
        "SupervisionInterval",
        "LevelInterval",
        "_overlaps",
        "_adjacent",
        "_intersection",
        "_union",
    ]
    breakpoint()
    df[collst].head(10)
    df.loc[df["_adjacent"], collst].head(10)
    return None


def test_pdintervalso(supdf, lvldf):
    df = pd.merge(supdf, lvldf, how="inner", left_index=True, right_index=True)
    durations = []
    durations.append(default_timer())
    df["_overlaps"] = df.apply(
        overlaps, axis=1, args=("SupervisionInterval", "LevelInterval")
    )
    durations.append(default_timer())
    df["_adjacent"] = df.apply(
        adjacent, axis=1, args=("SupervisionInterval", "LevelInterval")
    )
    durations.append(default_timer())
    df["_intersection"] = df.apply(
        intersection, axis=1, args=("SupervisionInterval", "LevelInterval")
    )
    durations.append(default_timer())
    df["_union"] = df.apply(
        union, axis=1, args=("SupervisionInterval", "LevelInterval")
    )
    durations.append(default_timer())
    durarray = np.diff(np.array(durations), n=1)
    methodlst = ["overlaps", "adjacent", "intersection", "union"]
    durationsdct = {m: v for m, v in zip(methodlst, durarray)}
    collst = [
        "SupervisionInterval",
        "LevelInterval",
        "_overlaps",
        "_adjacent",
        "_intersection",
        "_union",
    ]
    breakpoint()
    df[collst].head(10)
    df.loc[df["_adjacent"], collst].head(10)
    return None


def test_grpby_piso(sup2df, lvl2df):
    sdf = sup2df.reset_index().groupby("Id")
    ldf = lvl2df.reset_index().groupby("Id")
    dflst = []
    for id in sdf.groups:
        sgdf = sdf.get_group(id)
        try:
            lgdf = ldf.get_group(id)
        except KeyError:
            continue
        if sgdf.shape[0] and lgdf.shape[0]:
            # sgdf = sgdf.set_index("Interval")
            # lgdf = lgdf.set_index("Interval")
            # dflst.append(piso.join(sgdf, lgdf, how="inner", suffixes=["Sup", "Lvl"]))
            df = pd.merge(
                sgdf,
                lgdf,
                how="inner",
                left_index=True,
                right_index=True,
                suffixes=["Sup", "Lvl"],
            )
            breakpoint()
            df["_intersection"] = piso.intersection(
                df["IntervalSup"].array, df["IntervalLvl"].array
            )
            dflst.append(df)
    df = pd.concat(dflst)
    breakpoint()
    df.head()


def test_piso(sup2df, lvl2df):
    sdf = sup2df.reset_index().set_index("Id")
    ldf = lvl2df.reset_index().set_index("Id")
    df = pd.merge(
        sdf,
        ldf,
        how="inner",
        left_index=True,
        right_index=True,
        suffixes=["Sup", "Lvl"],
    )
    durations = []
    durations.append(default_timer())
    df["_intersection"] = df.apply(
        intersection_piso, axis=1, args=("IntervalSup", "IntervalLvl")
    )
    durations.append(default_timer())
    df["_union"] = df.apply(union_piso, axis=1, args=("IntervalSup", "IntervalLvl"))
    durations.append(default_timer())
    durarray = np.diff(np.array(durations), n=1)
    # methodlst = ["overlaps", "adjacent", "intersection", "union"]
    methodlst = ["intersection", "union"]
    durationsdct = {m: v for m, v in zip(methodlst, durarray)}
    breakpoint()
    durationsdct


def test_intervaloverlap(basedf):
    dtrng = pd.date_range("2015-01-01", "2016-01-01", freq="D")
    dtcols = ["Start_Date", "End_Date"]
    for col in dtcols:
        basedf[col] = basedf[col].apply(lambda x: pd.Timestamp(x))
    breakpoint()
    pass


# def test_dt64(basedf):
#     start = np.datetime64('2015-01-10')
#     end = np.datetime64('2016-07-28')
#     breakpoint()
