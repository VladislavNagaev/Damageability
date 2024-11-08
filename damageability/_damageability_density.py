import numpy as np
from numpy.typing import NDArray

import numba as nb

from pydantic import validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def damageability_density(
    damageability_array:NDArray,
    time_array:NDArray,
    start_cycle_indexes:NDArray,
    end_cycle_indexes:NDArray,
) -> NDArray:
    """
    Осуществляет расчет массива значений плотности повреждаемости.

    Args:
        damageability_array : ndarray
            Массив значений накопленной повреждаемости.
        time_array : ndarray
            Массив значений времени.
        start_cycle_indexes : ndarray
            Массив индексов начала выделенных полных циклов.
        end_cycle_indexes : ndarray
            Массив индексов оконачния выделенных полных циклов.
    """

    return _damageability_density(
        damageability_array=damageability_array,
        time_array=time_array,
        start_indexes=start_cycle_indexes,
        end_indexes=end_cycle_indexes,
    )

arg_types = (nb.float64[::1], nb.float64[::1], nb.int64[::1], nb.int64[::1])
ret_type = nb.float64[::1]
sig = ret_type(*arg_types)

@nb.jit(sig, nopython=True)
def _damageability_density(
    damageability_array:NDArray,
    time_array:NDArray,
    start_indexes:NDArray,
    end_indexes:NDArray,
) -> NDArray:

    # Инициализация массивов плотности повреждаемости
    damageability_density = np.full(fill_value=0.0, shape=time_array.shape[0]-1, dtype=np.float64,)

    # Проход по выделенным полным циклам
    for i in range(start_indexes.size):
        # Продолжительность реализации цикла
        cycle_time = round(time_array[end_indexes[i]] - time_array[start_indexes[i]], 10)
        # Заполнение массивов плотности повреждаемости
        damageability_density[start_indexes[i]:end_indexes[i]] += damageability_array[i] / cycle_time
    
    return damageability_density


