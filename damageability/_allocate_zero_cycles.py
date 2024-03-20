import numpy as np

from numpy.typing import NDArray

from pydantic import validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def allocate_zero_cycles(
    start_full_cycles:NDArray, 
    end_full_cycles:NDArray,
) -> NDArray:
    """
    Выполняет приведение массива полных циклов к отнулевым циклам.

    Пиведение к эквивалентному отнулевому циклу осуществляется в соотвествии с
    формулой И.А. Одинга.
    
        { √(2∙ampl∙max)           при mean >= 0
        { √2 ∙ (ampl+0.2∙mean)    при mean < 0 и max > 0
        { 0                       при max <= 0

    Args:
        start_full_cycles : ndarray
            Массив значений начала полных циклов.
        end_full_cycles : ndarray
            Массив значений конца полных циклов.
    
    """

    return _allocate_zero_cycles(
        start_full_cycles=start_full_cycles,
        end_full_cycles=end_full_cycles,
    )


def _allocate_zero_cycles(
    start_full_cycles:NDArray, 
    end_full_cycles:NDArray,
) -> NDArray:
    
    full_cycles = np.vstack((start_full_cycles, end_full_cycles))

    max_full_cycles = np.max(a=full_cycles, axis=0,)
    min_full_cycles = np.min(a=full_cycles, axis=0,)
    mean_full_cycles = np.mean(a=full_cycles, axis=0,)
    amplitude_full_cycles = np.divide( np.subtract(max_full_cycles, min_full_cycles), 2)

    zero_cycles = np.zeros(shape=full_cycles.shape[1], )

    indexes_1 = mean_full_cycles >= 0
    indexes_2 = np.logical_and( ~(mean_full_cycles >= 0), (max_full_cycles > 0) )

    values_1 = np.sqrt(2 * amplitude_full_cycles * max_full_cycles)
    values_2 = np.sqrt(2) * (amplitude_full_cycles + 0.2 * mean_full_cycles)

    zero_cycles[indexes_1] = values_1[indexes_1]
    zero_cycles[indexes_2] = values_2[indexes_2]

    return zero_cycles

