#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extended methods for Pandas Intervals."""

import pandas as pd
from typing import Callable, List, Tuple


class XInterval:
    """Extended methods for working with Pandas Intervals."""
    def __init__(self) -> None:
        pass

    def _check_interval_order(self,
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

    def _merge_df(self, *args, **kwargs) -> pd.DataFrame:
        return pd.merge(*args, **kwargs)

    def _apply(self, method: Callable, left: pd.DataFrame, right: pd.DataFrame, how: str, left_on: List[str], right_on: List[str], left_interval: str, right_interval: str, output_column: str="_output") -> pd.DataFrame:
        df = self._merge_df(left, right, how=how, left_on=left_on, right_on=right_on)
        df[output_column] = df.apply(method, axis=1, intervalcol_left=left_interval, intervalcol_right=right_interval)

    def overlaps(self, output_column: str="_overlaps", **kwargs) -> pd.DataFrame:
        return self._apply(self._overlaps, **kwargs)

    def _overlaps(self, row: pd.Series, intervalcol_left: str, intervalcol_right: str) -> bool:
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

    def _adjacent(self, row: pd.Series, intervalcol_left: str, intervalcol_right: str) -> bool:
        intervalcol_left, intervalcol_right = self._check_interval_order(
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

    def _intersection(self,
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


    def _union(self,
        row: pd.Series, intervalcol_left: str, intervalcol_right: str
    ) -> pd.IntervalDtype:
        intervalcol_left, intervalcol_right = self._check_interval_order(
            row, intervalcol_left, intervalcol_right
        )
        if row[intervalcol_left].overlaps(row[intervalcol_right]) or self._adjacent(
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
