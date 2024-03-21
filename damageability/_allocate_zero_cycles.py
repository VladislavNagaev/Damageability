import numpy as np
from numpy.typing import NDArray

import numba as nb

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


arg_types = (nb.float64[::1], nb.float64[::1])
ret_type = nb.float64[::1]
sig = ret_type(*arg_types)

@nb.jit(sig, nopython=True)
def _allocate_zero_cycles(
    start_full_cycles:NDArray[np.float64], 
    end_full_cycles:NDArray[np.float64],
) -> NDArray[np.float64]:

    zero_cycles = list()
    
    for start_cycle, end_cycle in zip(start_full_cycles, end_full_cycles):
        
        min_cycle = min(start_cycle, end_cycle)
        max_cycle = max(start_cycle, end_cycle)
        mean_cycle = (start_cycle+end_cycle)/2
        amplitude_cycle = (max_cycle-min_cycle)/2

        if mean_cycle >= 0:
            zero_cycle = (2*amplitude_cycle*max_cycle)**0.5
        elif mean_cycle<0 and max_cycle>0:
            zero_cycle = (2**0.5)*(amplitude_cycle+0.2*mean_cycle)
        else:
            zero_cycle = 0

        zero_cycles.append(zero_cycle)

    return np.array(zero_cycles)

