import numpy as np
from numpy.typing import NDArray

import numba as nb

from typing import Literal, Optional, TypeAlias, overload
from annotated_types import Annotated, Ge, Gt

from ._allocate_extremums import _allocate_extremums
from ._allocate_zero_cycles import _allocate_zero_cycles
from ._calculate_damageability import _calculate_damageability

from pydantic import validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def allocate_full_cycles(
    values:NDArray,
    result_type:Literal['indexes','values']='indexes',
    sort:bool=True,
    fcd:Optional[Annotated[float, Gt(0.0)]]=None,
    assume_extremums:bool=False,
    smoothing_value:Annotated[float, Ge(0.0)]=0.0,
) -> tuple[NDArray,NDArray]:
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
        extremum_indexes = _allocate_extremums(values, smoothing_value)
        # Выделение полных циклов
        start_cycle_indexes_, end_cycle_indexes_ = _allocate_full_cycles(values[extremum_indexes], sort=sort, fcd=fcd,)
        # Индексы полных циклов в исходном массиве данных
        start_cycle_indexes = extremum_indexes[start_cycle_indexes_]
        end_cycle_indexes = extremum_indexes[end_cycle_indexes_]

    else:

        # Выделение полных циклов
        start_cycle_indexes, end_cycle_indexes =  _allocate_full_cycles(values, sort=sort, fcd=fcd,)

    if result_type == 'indexes':
        return start_cycle_indexes, end_cycle_indexes
    elif result_type == 'values':
        return values[start_cycle_indexes], values[end_cycle_indexes]


full_cycles: TypeAlias = tuple[NDArray[np.int64],NDArray[np.int64]]

@overload
def _allocate_full_cycles(
    values:NDArray[np.float64], 
    fcd:Optional[Annotated[float, Gt(0.0)]]=...,
    sort:bool=...,
    raw_data:Literal[False]=...,
) -> full_cycles: ...

@overload
def _allocate_full_cycles(
    values:NDArray[np.float64], 
    fcd:Optional[Annotated[float, Gt(0.0)]]=...,
    sort:bool=...,
    raw_data:Literal[True]=...,
) -> tuple[full_cycles,full_cycles,full_cycles]: ...

def _allocate_full_cycles(
    values:NDArray[np.float64], 
    fcd:Optional[Annotated[float, Gt(0.0)]]=None,
    sort:bool=True,
    raw_data:bool=False,
) -> full_cycles|tuple[full_cycles,full_cycles,full_cycles]:

    # Массивы индексов точек начала и окончания полных циклов и массив индексов точек остатков
    # основной реализации методом полных циклов
    main_start_indexes, main_end_indexes, remainder_indexes = _main_full_cycles(values)
    # реализации из остатков
    remainder_start_indexes, remainder_end_indexes, remainder_indexes_ = \
    _remainder_full_cycles(values[remainder_indexes],fcd)
    
    # Обратная индексация реализации из остатков
    remainder_start_indexes = remainder_indexes[remainder_start_indexes]
    remainder_end_indexes = remainder_indexes[remainder_end_indexes]
    remainder_indexes = remainder_indexes[remainder_indexes_]
    
    # Выделение оставшегося нераспределенного цикла
    if remainder_indexes.size == 2:
        last_start_indexes, last_end_indexes = remainder_indexes[[0]], remainder_indexes[[1]]
    else:
        last_start_indexes, last_end_indexes = np.array([], np.int64), np.array([], np.int64)
    
    # Объединение массивов индексов выделенных циклов
    start_indexes = np.concatenate((main_start_indexes, remainder_start_indexes, last_start_indexes))
    end_indexes = np.concatenate((main_end_indexes, remainder_end_indexes, last_end_indexes))
    
    # Сортировка
    if sort:
        # Амплитуды выделенных циклов
        amplitudes = np.divide(np.abs(np.subtract(values[end_indexes], values[start_indexes])), 2)
        # Индексы сортировки циклов по возрастанию амплитуды
        sort_indexes = np.argsort(amplitudes)
        # Сортировка циклов по возрастанию амплитуды
        start_indexes = start_indexes[sort_indexes]
        end_indexes = end_indexes[sort_indexes]

    # Вернуть "сырые" данные
    if raw_data:
        return (
            (main_start_indexes, main_end_indexes), 
            (remainder_start_indexes, remainder_end_indexes,), 
            (last_start_indexes, last_start_indexes),
        )
    else:
        return start_indexes, end_indexes


def _main_full_cycles(values:NDArray[np.float64],) -> tuple[NDArray[np.int64],NDArray[np.int64],NDArray[np.int64]]:
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
    
    Returns:
        start_indexes : ndarray
            Массив индексов точек начала полных циклов.
        end_indexes : ndarray
            Массив индексов точек окончания полных циклов.
        remainder_indexes : ndarray
            Массив индексов точек остатков.
    """
    
    # Массивы индексов точек начала и окончания полных циклов
    start_indexes = np.array([], np.int64,)
    end_indexes = np.array([], np.int64,)
    # Массив индексов точек остатков
    remainder_indexes = np.arange(values.size, dtype=np.int64,)

    # Флаг обхода массива
    cycle = True
    
    while cycle:

        # Выполнить единичный обход массива
        traversed_start_indexes, traversed_end_indexes, traversed_remainder_indexes = \
        __traversal_array( values[remainder_indexes] )    

        # Полные циклы были выделены в последнем обходе массива
        cycle = not remainder_indexes.size == traversed_remainder_indexes.size

        # Сохранение выделенных индексов точек начала и окончания полных циклов единичного прохода
        start_indexes = np.concatenate((start_indexes, remainder_indexes[traversed_start_indexes]), dtype=np.int64,)
        end_indexes = np.concatenate((end_indexes, remainder_indexes[traversed_end_indexes]), dtype=np.int64,)
        # Обновление индексов точек остатков
        remainder_indexes = remainder_indexes[traversed_remainder_indexes]
        
    return start_indexes, end_indexes, remainder_indexes


arg_type = nb.float64[::1]
ret_type = nb.types.Tuple((nb.int64[::1], nb.int64[::1], nb.int64[::1]))
sig = ret_type(arg_type)

@nb.jit(sig, nopython=True)
def __traversal_array(values:NDArray[np.float64],) -> tuple[NDArray[np.int64],NDArray[np.int64],NDArray[np.int64]]:
    """Выполнить единичный обход массива, выделяя полные циклы."""
    
    # Массивы индексов точек начала и окончания полных циклов
    start_indexes = list()
    end_indexes = list()
    # Массив индексов точек остатков
    remainder_indexes = list()

    # Размер массива
    n = values.size
    
    # Индексы нулевой и первой точек потенциального цикла
    # (индексы второй и третьей точки образуются суммированием первой точки с единицей и двойкой сооответственно)
    i0, i1 = 0, 1

    while i1<=n-3:

        # Условие при котором 1 и 2 точки образуют полный цикл
        if (
            abs(values[i1]-values[i1+1])<=abs(values[i0  ]-values[i1  ])
            and
            abs(values[i1]-values[i1+1])<=abs(values[i1+1]-values[i1+2])
        ):
            start_indexes.append(i1)
            end_indexes.append(i1+1)
            i1+=2
        else:
            remainder_indexes.append(i0)
            i0, i1 = i1, i1+1

    # Обработка остатка
    remainder_indexes.append(i0)
    if i1==n-2:
        remainder_indexes.append(i1  )
        remainder_indexes.append(i1+1)

    return np.array(start_indexes, np.int64), np.array(end_indexes, np.int64), np.array(remainder_indexes, np.int64)


def _remainder_full_cycles(
    values:NDArray[np.float64], 
    fcd:Optional[Annotated[float, Gt(0.0)]]=None,
) -> tuple[NDArray[np.int64],NDArray[np.int64],NDArray[np.int64]]:
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
        fcd : float, optional
            Показатель степени кривой усталости материала (fatigue curve degree).
            Default to None
    
    Returns:
        start_indexes : ndarray
            Массив индексов точек начала полных циклов.
        end_indexes : ndarray
            Массив индексов точек окончания полных циклов.
        remainder_indexes : ndarray
            Массив индексов точек остатков.
    
    """
    
    # Индексы экстремумов
    max_indexes = np.where(np.equal(values, values.max()))[0]
    min_indexes = np.where(np.equal(values, values.min()))[0]
    
    if min_indexes.size>1:
        max_indexes = np.repeat(max_indexes, min_indexes.size)
    elif max_indexes.size>1:
        min_indexes = np.repeat(min_indexes, max_indexes.size)
    
    # Варианты реализации массивов индексов точек начала и окончания полных циклов
    start_indexes_implementations = list()
    end_indexes_implementations = list()
    # Варианты реализации значений накопленной повреждаемости
    damageability_implementations = list()
    
    # Массив индексов точек остатков
    remainder_indexes = np.arange(values.size, dtype=np.int64,)
    
    # Проход по вариантам комбинации экстремумов
    for max_index, min_index in zip(max_indexes, min_indexes):
    
        # Индексы начала и окончания цикла ЗВЗ
        start_aga_index = min(max_index, min_index)
        end_aga_index = max(max_index, min_index)
    
        # Индексы начала и окончания циклов
        start_indexes = remainder_indexes[start_aga_index%2::2]
        end_indexes = remainder_indexes[start_aga_index%2+1::2]
        
        # Количество циклов
        size = min(start_indexes.size, end_indexes.size)
        
        # Корректировка количества элементов
        start_indexes = start_indexes[:size]
        end_indexes = end_indexes[:size]
    
        # Массив отнулевых циклов 
        # прямой реализации обработки на повреждаемость
        zero_cycles_pos = _allocate_zero_cycles(values[start_indexes], values[end_indexes])
        # обратной реализации обработки на повреждаемость
        zero_cycles_neg = _allocate_zero_cycles(-values[start_indexes],-values[end_indexes])
    
        if fcd:
    
            # Накопленная повреждаемость оцениваемой реализации
            # прямой реализации обработки на повреждаемость
            damageability_pos = _calculate_damageability(zero_cycles_pos, fcd).sum()
            # обратной реализации обработки на повреждаемость
            damageability_neg = _calculate_damageability(zero_cycles_neg, fcd).sum()
    
        else:
            
            # Эквивалент накопленной повреждаемости оцениваемой реализации
            # прямой реализации обработки на повреждаемость
            damageability_pos = zero_cycles_pos.sum()
            # обратной реализации обработки на повреждаемость
            damageability_neg = zero_cycles_neg.sum()
    
        # Суммарное значение повреждаемости прямой и обратной реализации
        damageability = damageability_pos + damageability_neg
    
        # Сохранение вариантов реализации массивов индексов точек начала и окончания полных циклов
        start_indexes_implementations.append(start_indexes)
        end_indexes_implementations.append(end_indexes)
        # Сохранение вариантов реализации значений накопленной повреждаемости
        damageability_implementations.append(damageability)
    
    # Выбор варианта реализации
    implementation = np.argmax(damageability_implementations)
    
    # Выбор реализации индексов начала и окончания циклов
    start_indexes = start_indexes_implementations[implementation]
    end_indexes = end_indexes_implementations[implementation]
    
    # Обновление массива индексов точек остатков
    remainder_indexes = np.setdiff1d(remainder_indexes, np.concatenate((start_indexes, end_indexes)),)

    return start_indexes, end_indexes, remainder_indexes

