import numpy as np

from numpy.typing import NDArray
from annotated_types import Annotated, Gt

from pydantic import validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def calculate_damageability(
    zero_cycles:NDArray, 
    fcd:Annotated[float, Gt(0.0)]=4.0,
) -> NDArray:
    """
    Выполняет вычисление накопленной повреждаемости по значениям отнулевых циклов.

    Args:
        zero_cycles : ndarray
            Массив значений отнулевых циклов.
        fcd : float
            Показатель степени кривой усталости материала.
            Default to 4.0
    
    """
    
    return _calculate_damageability(zero_cycles, fcd)


def _calculate_damageability(
    zero_cycles:NDArray, 
    fcd:Annotated[float, Gt(0.0)],
) -> NDArray:    
    return np.power(zero_cycles, fcd)


