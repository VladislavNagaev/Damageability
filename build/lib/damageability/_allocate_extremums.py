import numpy as np

from numpy.typing import NDArray
from typing import Literal, Union, Tuple

from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def allocate_extremums(
    values:NDArray, 
    smoothing_value:float=0.0, 
    result_type:Literal['indexes','values','both']='values',
) -> Union[NDArray, Tuple[NDArray,NDArray]]:
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
    smoothing_value:float, 
    result_type:Literal['indexes','values','both'],
) -> Union[NDArray, Tuple[NDArray,NDArray]]:

    # Массив значений и индексов
    values = values.copy()
    indexes = np.arange(stop=values.size,)
    # Удаление дублирующихся элементов (остается первый уникальный элемент последовательности)
    indexes = indexes[~np.equal(np.append(np.nan, np.roll(values[indexes], shift=1,)[1:]), values)]
    # Направление движения цикла на текущей точке относительно предыдущей
    sign = np.append(0,np.sign(np.subtract(values[indexes],np.roll(values[indexes],shift=1,)))[1:])
    # Удаление промежуточных точек в нисходящих / восходящих циклах
    indexes = indexes[~np.equal(np.roll(sign, shift=-1, ), sign)]
    
    # Сглаживание
    if smoothing_value > 0:
        indexes = indexes[__perform_smoothing(values=values[indexes], smoothing_value=smoothing_value)]
    
    if result_type == 'indexes':
        return indexes
    elif result_type == 'values':
        return values[indexes]
    elif result_type == 'both':
        return values[indexes], indexes
    else:
        raise ValueError()

def __perform_smoothing(
    values:NDArray, 
    smoothing_value:float, 
) -> NDArray:

    # Инициализация массива индексов с первым элементом
    indexes = [0, ]
    # Направление движения цикла на текущей точке относительно предыдущей растет 
    rising = (np.append(0, np.diff(values)) > 0).tolist()
    # Текущий пик
    current = None
    
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
                
    # Преобразование списка в массив
    return np.array(indexes)


