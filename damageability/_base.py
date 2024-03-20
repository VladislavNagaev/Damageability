import numpy as np

from ._allocate_extremums import _allocate_extremums
from ._allocate_full_cycles import _allocate_full_cycles
from ._allocate_zero_cycles import _allocate_zero_cycles
from ._calculate_damageability import _calculate_damageability
from .Damageability import Damageability

from numpy.typing import NDArray
from typing import Any
from annotated_types import Annotated, Ge, Gt

from pydantic import validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def damageability(
    values:NDArray, 
    smoothing_value:Annotated[float, Ge(0.0)]=0.0,
    fcd:Annotated[float, Gt(0.0)]=4.0,
    **kwargs,
) -> Damageability:
    """
    Выполняет вычисление накопленной повреждаемости.

        В процессе обработки происходит выделение экстремумов и сглаживание.
        По выделенным значениям экстремумов осуществляется выделение полных циклов
    методом полных циклов.
        Полученные циклы приводятся к эквивалентным отнулевым циклам с использованием
    формулы И.А. Одинга.
        Значения полученныъ эквивалентных циклов приводятся к эквивалентной повреждаемости 
    отнулевого цикла.
        Аналогичная процедура проводится для обратной реализации (отрицательные исходные значения).
    
    Args:
        values : ndarray
            Массив исходных значений.
        time_array : ndarray, optional
            Массив значений времени. Если не задано - будет установлен стандартный
            массив значений с интервалом единица.
            Default to None
        smoothing_value : int
            Уровень сглаживания при выделении экстремумов.
            Default to 0.0
        fcd : float
            Показатель степени кривой усталости материала (fatigue curve degree).
            Default to 4.0
    
    Returns:
        Damageability : type
            Экземпляр класса повреждаемости.

    """

    return Damageability(**_damageability(
        values=values,
        smoothing_value=smoothing_value,
        fcd=fcd,
        sort=True,
    ))


def _damageability(
    values:NDArray, 
    smoothing_value:Annotated[float, Ge(0.0)],
    fcd:Annotated[float, Gt(0.0)],
    sort:bool,
) -> dict[str,Any]:

    # Выделение экстремумов
    extremum_indexes = _allocate_extremums(
        values=values,
        smoothing_value=smoothing_value,
        result_type='indexes',
    )

    # Выделение полных циклов
    start_cycle_indexes_, end_cycle_indexes_ = _allocate_full_cycles(
        values=values[extremum_indexes], 
        result_type='indexes',
        sort=sort, 
        fcd=fcd,
    )

    # Индексы полных циклов в исходном массиве данных
    start_cycle_indexes = extremum_indexes[start_cycle_indexes_]
    end_cycle_indexes = extremum_indexes[end_cycle_indexes_]

    # Массив отнулевых циклов 
    # прямой реализации обработки на повреждаемость
    zero_cycles_positive = _allocate_zero_cycles(
        start_full_cycles=values[start_cycle_indexes], 
        end_full_cycles=values[end_cycle_indexes],
    )
    # обратной реализации обработки на повреждаемость
    zero_cycles_negative = _allocate_zero_cycles(
        start_full_cycles=np.negative(values[start_cycle_indexes]), 
        end_full_cycles=np.negative(values[end_cycle_indexes]),
    )

    # Массив значений накопленной повреждаемости
    # прямой реализации обработки на повреждаемость
    damageability_array_positive = _calculate_damageability(
        zero_cycles=zero_cycles_positive,
        fcd=fcd,
        result_type='array',
    )
    # обратной реализации обработки на повреждаемость
    damageability_array_negative = _calculate_damageability(
        zero_cycles=zero_cycles_negative,
        fcd=fcd,
        result_type='array',
    )

    return dict(
        values=values,
        smoothing_value=smoothing_value,
        fcd=fcd,
        extremum_indexes=extremum_indexes,
        start_cycle_indexes=start_cycle_indexes,
        end_cycle_indexes=end_cycle_indexes,
        zero_cycles_positive=zero_cycles_positive,
        zero_cycles_negative=zero_cycles_negative,
        damageability_array_positive=damageability_array_positive,
        damageability_array_negative=damageability_array_negative,
    )

