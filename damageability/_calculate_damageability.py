import numpy as np

from numpy.typing import NDArray
from typing import Literal
from annotated_types import Annotated, Gt

from pydantic import validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def calculate_damageability(
    zero_cycles:NDArray, 
    fcd:Annotated[float, Gt(0.0)]=4.0,
    result_type:Literal['array','value']='value',
) -> float:
    """
    Выполняет вычисление накопленной повреждаемости по значениям отнулевых циклов.

    
    Args:
        zero_cycles : ndarray
            Массив значений отнулевых циклов.
        fcd : float
            Показатель степени кривой усталости материала.
            Default to 4.0
        result_type : 'array' or 'value'
            Тип возвращаемого результата:
            'array' - массив значений повреждаемости каждого цикла;
            'value' - суммарное значение накопленной повреждаемости.
            Default to 'value'
    
    """
    
    return _calculate_damageability(
        zero_cycles=zero_cycles,
        fcd=fcd,
        result_type=result_type,
    )


def _calculate_damageability(
    zero_cycles:NDArray, 
    fcd:Annotated[float, Gt(0.0)],
    result_type:Literal['array','value'],
) -> float:
    
    damageability = np.power(zero_cycles, fcd)

    if result_type == 'array':
        return damageability
    elif result_type == 'value':
        return np.sum(damageability)

