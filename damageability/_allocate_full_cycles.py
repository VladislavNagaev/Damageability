import numpy as np

from numpy.typing import NDArray
from typing import Tuple, Literal, Optional, Union

from ._allocate_extremums import _allocate_extremums
from ._allocate_zero_cycles import _allocate_zero_cycles
from ._calculate_damageability import _calculate_damageability

from pydantic import validate_arguments


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def allocate_full_cycles(
    values:NDArray,
    result_type:Literal['indexes','values']='indexes',
    sort:bool=True,
    fcd:Optional[float]=None,
    assume_extremums:bool=False,
    smoothing_value:float=0.0,
) -> Tuple[NDArray,NDArray]:
    """
    Осуществляет выделение полных циклов из массива данных.

    Более подробное описание приведено в книге Стопкевич В.Г. Прочностная надежность боевых летательных аппаратов, 1991, стр. 29-36
    
    Args:
        values : ndarray
            Массив обрабатываемых значений. Значения должны быть массивом экстремумов.
        result_type : 'indexes' or 'values'
            Тип возвращаемого результата:
            'indexes' - массив индексов значений выделенных циклов;
            'values' - массив абсолютных значений выделенных циклов.
            Default to 'indexes'
        sort : bool
            Выполнять сортировку выделенных полных циклов по возрастанию амплитуды
            Default to True
        fcd : float, optional
            Показатель степени кривой усталости материала (fatigue curve degree).
            Используется для вычисления накопленной повреждаемости при выделении полных циклов
            из массива остатков, в котором обнаружено два цикла ЗВЗ. 
            Если не задано - оценка будет производиться по значениям амплитуд, что 
            потенциально может дать незначительное отклонение в точности выделения 
            полных циклов. 
            На практике данный показатель не оказывает сколько-либо значимых изменений, но
            усложняет процесс расчета.
            Default to None
        assume_extremums : bool, optional
            If True, the input array is assumed to be unique.
            Default to False
        smoothing_value : int
            Уровень сглаживания при выделении экстремумов.
            Используется в случае, если параметр assume_extremums установлен False.
            Default to 0.0

    Returns:
        start_cycle_indexes : ndarray
            Массив индексов начала выделенных полных циклов
        end_cycle_indexes : ndarray
            Массив индексов оконачния выделенных полных циклов
    
    """
    
    if assume_extremums == False:

        # Выделение экстремумов
        extremum_indexes = _allocate_extremums(
            values=values,
            smoothing_value=smoothing_value,
            result_type='indexes',
        )
    
        start_cycle_indexes_, end_cycle_indexes_ = _allocate_full_cycles(
            values=values[extremum_indexes],
            result_type='indexes',
            sort=sort,
            fcd=fcd,
        )

        # Индексы полных циклов в исходном массиве данных
        start_cycle_indexes = extremum_indexes[start_cycle_indexes_]
        end_cycle_indexes = extremum_indexes[end_cycle_indexes_]

        if result_type == 'indexes':
            return start_cycle_indexes, end_cycle_indexes
        elif result_type == 'values':
            return values[start_cycle_indexes], values[end_cycle_indexes]

    else:

        return _allocate_full_cycles(
            values=values,
            result_type=result_type,
            sort=sort,
            fcd=fcd,
        )


def _allocate_full_cycles(
    values:NDArray,
    result_type:Literal['indexes','values'],
    sort:bool,
    fcd:Optional[float],
    raw_data:bool=False,
) -> Union[Tuple[NDArray,NDArray], Tuple[Tuple[NDArray,NDArray],Tuple[NDArray,NDArray],Tuple[NDArray,NDArray]]]:
    
    # Массив индексов обрабатываемых значений
    indexes = np.arange(stop=values.size,)

    # Выделение полных циклов методом полных циклов
    start_indexes_main, end_indexes_main, indexes = __allocate_main_cycles(
        values=values, 
        indexes=indexes,
    )
    # Выделение полных циклов из массива остатков
    start_indexes_remainder, end_indexes_remainder, indexes = __allocate_remainder_cycles(
        values=values, 
        indexes=indexes,
        fcd=fcd,
    )
    # Выделение оставшегося нераспределенного цикла
    if indexes.size == 2:
        start_indexes_last, end_indexes_last = indexes[[0]], indexes[[1]]
    else:
        start_indexes_last, end_indexes_last = np.array([], dtype=np.int64, ), np.array([], dtype=np.int64, )

    # Вернуть "сырые" данные
    if raw_data:
        return (
            (start_indexes_main, end_indexes_main), 
            (start_indexes_remainder, end_indexes_remainder,), 
            (start_indexes_last, end_indexes_last),
        )

    # Объединение массивов индексов выделенных циклов
    start_cycle_indexes = np.concatenate(( start_indexes_main, start_indexes_remainder, start_indexes_last, ))
    end_cycle_indexes = np.concatenate(( end_indexes_main, end_indexes_remainder, end_indexes_last ))

    if sort:

        # Амплитуды выделенных циклов
        amplitudes = np.divide(np.abs(np.subtract(
            values[end_cycle_indexes], 
            values[start_cycle_indexes],
        )), 2)
        
        # Индексы сортировки циклов по возрастанию амплитуды
        sort_indexes = np.argsort(amplitudes)

        # Сортировка циклов по возрастанию амплитуды
        start_cycle_indexes = start_cycle_indexes[sort_indexes]
        end_cycle_indexes = end_cycle_indexes[sort_indexes]

    if result_type == 'indexes':
        return start_cycle_indexes, end_cycle_indexes
    elif result_type == 'values':
        return values[start_cycle_indexes], values[end_cycle_indexes]
    else:
        raise ValueError('unknown result_type')
        

def __allocate_main_cycles(
    values:NDArray, 
    indexes:NDArray,
) -> Tuple[NDArray,NDArray,NDArray]:
    """
    Выполняет выделение полных циклов методом полных циклов. 
    Возвращает массивы индексов начала и окончания выделенных циклов и 
    массив индексов остатка
    
    В реализации отыскиваются какие-либо четыре подряд идущих экстремума
    (начальная и конечная точки реализации "засчитываются" в качестве экстремумов)
    с величинами Э_i, Э_(i+1), Э_(i+2), Э_(i+3), для которых осуществляются 
    совместные неравенства
        | Э_(i+1) - Э_(i+2) | <= | Э_i     - Э_(i+1) |
        | Э_(i+1) - Э_(i+2) | <= | Э_(i+2) - Э_(i+3) |
    "Внутренний" цикл с экстремумами Э_(i+1) и Э_(i+2) признакется полным циклом,
    его характеристики (т.е. величины максимума и минимума) запоминаются, а сами
    экстремумы Э_(i+1) и Э_(i+2) из реализации исключаются. С остающейся реализацией
    многократно выполняется та же процедура, при которой, как правило, представляется 
    необходимым двигаться по реализации (и остающейся её части) вперед и назад.
    Обработка таким образом продолжается до тех пор, пока либо вся реализация 
    не будет переведена в совокупность полных циклов, либо пока не сохранится 
    остаток, имеющий вид последовательно расходящихся и (или) сходящихся 
    амплитуд колебаний, для которого изложенный алгоритм циклообразования 
    "перестает работать".
    
    Args:
        values : ndarray
            Массив исходных значений. Значения должны быть массивом экстремумов.
        indexes : ndarray
            Массив индексов значений, подлежащих обработке.
    
    Returns:
        start_full_cycle : ndarray
            Массив значений начала полных циклов.
        end_full_cycle : ndarray
            Массив значений конца полных циклов.
        indexes : ndarray
            Массив индексов значений, подлежащих обработке.
        
    """

    # Пустые массивы для сохранения результатов
    start_cycle_indexes = np.array([], dtype=np.int64, )
    end_cycle_indexes = np.array([], dtype=np.int64, )

    while True:

        # Участки, на которых выполняется условие
        cycle_positions = np.logical_and(
            np.less_equal(
                np.abs(np.subtract(values[indexes][1:-2],values[indexes][2:-1])),
                np.abs(np.subtract(values[indexes][ :-3],values[indexes][1:-2])),
            ),
            np.less_equal(
                np.abs(np.subtract(values[indexes][1:-2],values[indexes][2:-1])),
                np.abs(np.subtract(values[indexes][2:-1],values[indexes][3:  ])),
            ),
        )

        # Прерывание цикла в случае отсуствия циклов в массиве
        if not np.any(cycle_positions):
            break

        # Коррекитровка участков, на которых выполняется условие
        cycle_positions = __remove_intersection_cycles(
            cycle_positions=cycle_positions, 
            values=values[indexes],
        )

        # Позиции выделенных циклов
        start_cycle_positions = np.add(np.where(cycle_positions)[0], 1)
        end_cycle_positions = np.add(np.where(cycle_positions)[0], 2)

        # Индексы выделенных циклов
        start_cycle_indexes = np.concatenate(( start_cycle_indexes, indexes[start_cycle_positions], ))
        end_cycle_indexes = np.concatenate(( end_cycle_indexes, indexes[end_cycle_positions], ))

        # Позиции элементов, подлежащих удалению из рабочих массивов
        remove_positions = np.concatenate((start_cycle_positions, end_cycle_positions,))

        # Обновление рабочих массивов
        indexes = np.delete(indexes, remove_positions)

    return start_cycle_indexes, end_cycle_indexes, indexes


def __remove_intersection_cycles(
    cycle_positions:NDArray, 
    values:NDArray, 
) -> NDArray:
    
    # Группы выделяемых циклов
    cycle_groups = np.cumsum(np.append(True, np.diff(a=cycle_positions,) ))
    # Количества циклов в группах
    unique_groups, unique_counts = np.unique(ar=cycle_groups, return_counts=True)
    # Статус выделения циклов в группах
    unique_cycle = cycle_positions[np.searchsorted(a=cycle_groups, v=unique_groups, side='left', )]
    # Корректировка крайних значений статуса выделения циклов в группах
    unique_cycle[[0,-1]] = True
    # Группы разделители
    groups_split = np.logical_and(np.greater_equal(unique_counts, 2), ~unique_cycle)
    # Циклы разделители
    split_cycle = np.repeat(a=groups_split, repeats=unique_counts, )
    # Список индексов групп циклов
    indexes_split = np.split(
        ary=np.arange(stop=cycle_positions.size), 
        indices_or_sections=np.add(np.where(np.diff(a=split_cycle,))[0], 1),
    )
    # Список амплитуд групп циклов
    amplitudes_split = np.split(
        ary=np.multiply(np.abs(np.diff(values))[1:-1], cycle_positions), 
        indices_or_sections=np.add(np.where(np.diff(a=split_cycle,))[0], 1),
    )
    # Индексы участков, на которых выполняется условие (корректировка)
    cycle_positions_indexes = [
        index[np.nonzero(amplitude)[0][np.argmin(amplitude[np.nonzero(amplitude)[0]])]]
        for index, amplitude in zip(indexes_split[::2], amplitudes_split[::2])
    ]
    # Инициализация отрицательного массива участков
    cycle_positions = np.full(fill_value=False, shape=cycle_positions.shape, dtype=bool, )
    # аполнение участков, на которых выполняется условие (корректировка)
    cycle_positions[cycle_positions_indexes] = True
    
    return cycle_positions


def __allocate_remainder_cycles(
    values:NDArray, 
    indexes:NDArray,
    fcd:Optional[float],
) -> Tuple[NDArray,NDArray,NDArray]:
    """
    Выполняет выделение полных циклов из массива остатков после выделения
    методом полных циклов.
    Остаток может представлять собой монотонно восходящую последовательность /
    монотонно нисходящую последовательность / последовательность одновременно 
    с левой стороны от цикла ЗВЗ восходящей, а с правой стороны от цикла ЗВЗ нисходящей.
    При этом любая из трех последовательностей может включать в себя от одного
    до двух полуциклов ЗВЗ.
    Выделение "полных полуциклов" из остатка осуществляется следующим образом:
    - если полуцикл ЗВЗ в последовательности один, то он считается центральным полуциклом,
    а оставшиеся полуциклы выделяются "через один" слева и справа от него, включая его самого;
    - если полуциклов ЗВЗ два, то выделение полуциклов осуществляется аналогично
    случаю с одним полуциклом ЗВЗ для двух реализаций: когда центральным полуциклом признается
    первый полуцикл ЗВЗ и когда второй. Для обеих реализаций проводится сравнение суммарных 
    значений амплитуд выделенных "полных циклов" и выбирается та реализация, где 
    суммарная повреждаемость больше.

    Args:
        values : ndarray
            Массив исходных значений.
        indexes : ndarray
            Массив индексов значений, подлежащих обработке.
        fcd : float
            Показатель степени кривой усталости материала (fatigue curve degree).
    
    Returns:
        start_full_cycle : ndarray
            Массив значений начала полных циклов.
        end_full_cycle : ndarray
            Массив значений конца полных циклов.
        indexes : ndarray
            Массив индексов значений, подлежащих обработке.
    
    """
    
    # Индексы экстремумов
    max_indexes = np.where(values[indexes] == np.max(values[indexes]))[0]
    min_indexes = np.where(values[indexes] == np.min(values[indexes]))[0]

    if (max_indexes.size == 1) and (min_indexes.size > 1):
        max_indexes = np.repeat(a=max_indexes, repeats=min_indexes.size,)
    elif (max_indexes.size > 1) and (min_indexes.size == 1):
        min_indexes = np.repeat(a=min_indexes, repeats=max_indexes.size,)
    elif (max_indexes.size == 1) and (min_indexes.size == 1):
        pass
    else:
        raise ValueError()

    # Инициализация списков реализаций
    start_cycle_indexes_list = list()
    end_cycle_indexes_list = list()
    damageability_list = list()

    # Проход по вариантам комбинации экстремумов
    for max_index, min_index in zip(max_indexes, min_indexes):

        # Индексы начала и окончания цикла ЗВЗ
        start_cycle_aga_index = min(max_index, min_index)
        end_cycle_aga_index = max(max_index, min_index)

        # Индексы начала и окончания циклов
        start_cycle_indexes = indexes[start_cycle_aga_index%2::2]
        end_cycle_indexes = indexes[start_cycle_aga_index%2+1::2]
        # Количество циклов
        size = min(start_cycle_indexes.size, end_cycle_indexes.size)
        # Корректировка количества элементов
        start_cycle_indexes = start_cycle_indexes[:size]
        end_cycle_indexes = end_cycle_indexes[:size]

        # Массив отнулевых циклов 
        # прямой реализации обработки на повреждаемость
        zero_cycles_pos = _allocate_zero_cycles(
            start_full_cycles=values[start_cycle_indexes], 
            end_full_cycles=values[end_cycle_indexes],
        )
        # обратной реализации обработки на повреждаемость
        zero_cycles_neg = _allocate_zero_cycles(
            start_full_cycles=np.negative(values[start_cycle_indexes]),
            end_full_cycles=np.negative(values[end_cycle_indexes]),
        )

        if fcd is not None:

            # Накопленная повреждаемость оцениваемой реализации
            # прямой реализации обработки на повреждаемость
            damageability_value_pos = _calculate_damageability(
                zero_cycles=zero_cycles_pos,
                fcd=fcd,
                result_type='value',
            )
            # обратной реализации обработки на повреждаемость
            damageability_value_neg = _calculate_damageability(
                zero_cycles=zero_cycles_neg,
                fcd=fcd,
                result_type='value',
            )

        else:
            
            # Эквивалент накопленной повреждаемости оцениваемой реализации
            # прямой реализации обработки на повреждаемость
            damageability_value_pos = np.sum(zero_cycles_pos)
            # обратной реализации обработки на повреждаемость
            damageability_value_neg = np.sum(zero_cycles_neg)

        # Суммарное значение повреждаемости прямой и обратной реализации
        damageability_value = damageability_value_pos + damageability_value_neg

        # Сохранение элементов реализаций
        start_cycle_indexes_list.append(start_cycle_indexes)
        end_cycle_indexes_list.append(end_cycle_indexes)
        damageability_list.append(damageability_value)

    # Выбор варианта реализации
    implementation_variant = np.argmax(damageability_list)

    # Выбор реализации индексов начала и окончания циклов
    start_cycle_indexes = start_cycle_indexes_list[implementation_variant]
    end_cycle_indexes = end_cycle_indexes_list[implementation_variant]

    # Обновление массива остаточных индексов
    indexes = np.setdiff1d(ar1=indexes, ar2=np.concatenate((start_cycle_indexes, end_cycle_indexes)),)

    return start_cycle_indexes, end_cycle_indexes, indexes


