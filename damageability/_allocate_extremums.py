import numpy as np
from numpy import int64
from numba import jit

from numpy.typing import NDArray
from typing import Literal, overload
from annotated_types import Annotated, Ge


from pydantic import validate_call


@overload
def allocate_extremums(
    values:NDArray, 
    smoothing_value:Annotated[float, Ge(0.0)]=..., 
    result_type:Literal['indexes']=...,
) -> NDArray: ...

@overload
def allocate_extremums(
    values:NDArray, 
    smoothing_value:Annotated[float, Ge(0.0)]=..., 
    result_type:Literal['values']=...,
) -> NDArray: ...

@overload
def allocate_extremums(
    values:NDArray, 
    smoothing_value:Annotated[float, Ge(0.0)]=..., 
    result_type:Literal['both']=...,
) -> tuple[NDArray,NDArray]: ...

@validate_call(config=dict(arbitrary_types_allowed=True))
def allocate_extremums(
    values:NDArray, 
    smoothing_value:Annotated[float, Ge(0.0)]=0.0, 
    result_type:Literal['indexes','values','both']='values',
) -> NDArray|tuple[NDArray,NDArray]:
    """
    Осуществляет выделение экстремумов.

    Args:
        values : ndarray
            Массив исходных значений.
        smoothing_value : int
            Уровень сглаживания при выделении экстремумов.
            Default to 0.0
        result_type : str
            Тип возвращаемого результата:
            'indexes' - массив индексов значений экстремумов в исходном массиве;
            'values' - массив значений экстремумов;
            'both' - массив значений и массив индексов.
            Default to 'values'

    """

    return _allocate_extremums(
        values=values,
        smoothing_value=smoothing_value,
        result_type=result_type,
    )


def _allocate_extremums(
    values:NDArray, 
    smoothing_value:Annotated[float, Ge(0.0)], 
    result_type:Literal['indexes','values','both'],
) -> NDArray|tuple[NDArray,NDArray]:

    indexes = __allocate_extremums(values, smoothing_value)
    
    if result_type == 'indexes':
        return indexes
    elif result_type == 'values':
        return values[indexes]
    elif result_type == 'both':
        return values[indexes], indexes
    else:
        raise ValueError()


@jit(nopython=True)
def __allocate_extremums(
    values:NDArray, 
    smoothing_value:Annotated[float, Ge(0.0)], 
) -> NDArray[int64]:

    # Инициализация массива индексов с первым элементом
    indexes = [0,]
    # Направление движения цикла на текущей точке относительно предыдущей растет 
    rising = np.greater(np.concatenate((np.array([0]), np.diff(values),)),0)
    # Текущий пик
    current = np.nan
    
    # Проход по всем элементам 
    for i in range(1,values.size):
        if not (
            (not rising[i] and (values[i] >= values[indexes[-1]]))
            or
            (rising[i] and (values[i] <= values[indexes[-1]]))
        ):
            if not rising[i] == current:
                if (abs(values[i] - values[indexes[-1]]) > smoothing_value):
                    current = rising[i]
                    indexes.append(i)
            else:
                indexes[-1] = i
    
    return np.array(indexes, dtype=int64,)

