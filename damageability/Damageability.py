import numpy as np

import warnings

from ._damageability_density import _damageability_density
from ._damageability_density_plot import _damageability_density_plot

from numpy.typing import NDArray
from matplotlib.figure import Figure
from typing import Optional, Any, TypedDict
from annotated_types import Annotated, Gt

from pydantic import validate_call


class DamageabilityDict(TypedDict):
    """Результирующий словарь значений повреждаемости."""
    damageability_positive_value: float
    damageability_negative_value: float
    damageability_aga_positive_value: float
    damageability_aga_negative_value: float
    equivalent_positive_value: float
    equivalent_negative_value: float
    equivalent_aga_positive_value: float
    equivalent_aga_negative_value: float


class Damageability:
    """
    Экземпляр класса повреждаемости.

    Parameters:
        extremum_values : ndarray
            Массивы значений экстремумов.
        extremum_indexes : ndarray
            Массивы индексов значений экстремумов.
        cycle_values : tuple of ndarray
            Массивы значений начала и конца полных циклов.
        cycle_indexes : tuple of ndarray
            Массивы индексов значений начала и конца полных циклов.
        zero_cycles_positive : ndarray
            Массив отнулевых циклов прямой реализации обработки на повреждаемость.
        zero_cycles_negative : ndarray
            Массив отнулевых циклов обратной реализации обработки на повреждаемость.
        damageability_array_positive : ndarray
            Массив значений накопленной повреждаемости прямой реализации обработки на повреждаемость.
        damageability_array_negative : ndarray
            Массив значений накопленной повреждаемости обратной реализации обработки на повреждаемость.
        damageability_full_positive : flaot
            Накопленное значение повреждаемости прямой реализации обработки на повреждаемость.
        damageability_full_negative : float
            Накопленное значение повреждаемости обратной реализации обработки на повреждаемость.
        damageability_aga_positive : float
            Повреждаемость цикла земля-воздух-земля прямой реализации обработки на повреждаемость.
        damageability_aga_negative : float
            Повреждаемость цикла земля-воздух-земля обратной реализации обработки на повреждаемость.
        equivalent_full_positive : float
            Эквивалентное значение накопленной повреждаемости прямой реализации обработки на повреждаемость.
        equivalent_full_negative : float
            Эквивалентное значение накопленной повреждаемости обратной реализации обработки на повреждаемость.
        equivalent_aga_positive : float
            Эквивалентное значение повреждаемости цикла земля-воздух-земля прямой реализации обработки на повреждаемость.
        equivalent_aga_negative : float
            Эквивалентное значение повреждаемости цикла земля-воздух-земля обратной реализации обработки на повреждаемость.
        
    Attributes:
        damageability_density : tuple of ndarray
            Массив значений плотности повреждаемости прямой и обратной реализаций обработки на повреждаемость.
        damageability_density_plot : matplotlib figure
            График плотности повреждаемости.

    """

    def __init__(
        self,
        values:NDArray, 
        smoothing_value:float,
        fcd:float,
        extremum_indexes:NDArray,
        start_cycle_indexes:NDArray,
        end_cycle_indexes:NDArray,
        zero_cycles_positive:NDArray,
        zero_cycles_negative:NDArray,
        damageability_array_positive:NDArray,
        damageability_array_negative:NDArray,
        **kwargs,
    ) -> None:
        
        # Инициализация параметров
        self._values = values
        self._smoothing_value = smoothing_value
        self._fcd = fcd
        self._extremum_indexes = extremum_indexes
        self._start_cycle_indexes = start_cycle_indexes
        self._end_cycle_indexes = end_cycle_indexes
        self._zero_cycles_positive = zero_cycles_positive
        self._zero_cycles_negative = zero_cycles_negative
        self._damageability_array_positive = damageability_array_positive
        self._damageability_array_negative = damageability_array_negative


    @property
    def _init_data(self) -> dict[str,Any]:
        return dict(
            _values = self._values, 
            _smoothing_value = self._smoothing_value,
            _fcd = self._fcd,
            _extremum_indexes = self._extremum_indexes,
            _start_cycle_indexes = self._start_cycle_indexes,
            _end_cycle_indexes = self._end_cycle_indexes,
            _zero_cycles_positive = self._zero_cycles_positive,
            _zero_cycles_negative = self._zero_cycles_negative,
            _damageability_array_positive = self._damageability_array_positive,
            _damageability_array_negative = self._damageability_array_negative,
        )
    

    @property
    def extremum_values(self,) -> NDArray:
        """Массивы значений экстремумов."""
        return self._values[self._extremum_indexes]


    @property
    def extremum_indexes(self,) -> NDArray:
        """Массивы индексов значений экстремумов."""
        return self._extremum_indexes
    

    @property
    def cycle_values(self,) -> tuple[NDArray,NDArray]:
        """Массивы значений начала и конца полных циклов."""
        return self._values[self._start_cycle_indexes], self._values[self._end_cycle_indexes]


    @property
    def cycle_indexes(self,) -> tuple[NDArray,NDArray]:
        """Массивы индексов значений начала и конца полных циклов."""
        return self._start_cycle_indexes, self._end_cycle_indexes


    @property
    def zero_cycles_positive(self,) -> NDArray:
        """Массив отнулевых циклов прямой реализации обработки на повреждаемость."""
        return self._zero_cycles_positive


    @property
    def zero_cycles_negative(self,) -> NDArray:
        """Массив отнулевых циклов обратной реализации обработки на повреждаемость."""
        return self._zero_cycles_negative


    @property
    def damageability_array_positive(self,) -> NDArray:
        """Массив значений накопленной повреждаемости прямой реализации обработки на повреждаемость."""
        return self._damageability_array_positive


    @property
    def damageability_array_negative(self,) -> NDArray:
        """Массив значений накопленной повреждаемости обратной реализации обработки на повреждаемость."""
        return self._damageability_array_negative


    @property
    def damageability_full_positive(self,) -> float:
        """Накопленное значение повреждаемости прямой реализации обработки на повреждаемость."""
        return float(np.sum(self._damageability_array_positive))


    @property
    def damageability_full_negative(self,) -> float:
        """Накопленное значение повреждаемости обратной реализации обработки на повреждаемость."""
        return float(np.sum(self._damageability_array_negative))


    @property
    def damageability_aga_positive(self,) -> float:
        """Повреждаемость цикла земля-воздух-земля прямой реализации обработки на повреждаемость."""
        return float(self._damageability_array_positive[-1])

    @property
    def damageability_aga_negative(self,) -> float:
        """Повреждаемость цикла земля-воздух-земля обратной реализации обработки на повреждаемость."""
        return float(self._damageability_array_negative[-1])


    @property
    def equivalent_full_positive(self,) -> float:
        """Эквивалентное значение накопленной повреждаемости прямой реализации обработки на повреждаемость."""
        return float(self.damageability_full_positive**(1/self._fcd))


    @property
    def equivalent_full_negative(self,) -> float:
        """Эквивалентное значение накопленной повреждаемости обратной реализации обработки на повреждаемость."""
        return float(self.damageability_full_negative**(1/self._fcd))


    @property
    def equivalent_aga_positive(self,) -> float:
        """Эквивалентное значение повреждаемости цикла земля-воздух-земля прямой реализации обработки на повреждаемость."""
        return float(self.damageability_aga_positive**(1/self._fcd))


    @property
    def equivalent_aga_negative(self,) -> float:
        """Эквивалентное значение повреждаемости цикла земля-воздух-земля обратной реализации обработки на повреждаемость."""
        return float(self.damageability_aga_negative**(1/self._fcd))

    @property
    def damageability_dict(self,) -> DamageabilityDict:
        """Результирующий словарь значений повреждаемости."""
        return {
            'damageability_positive_value': self.damageability_full_positive,
            'damageability_negative_value': self.damageability_full_negative,
            'damageability_aga_positive_value': self.damageability_aga_positive,
            'damageability_aga_negative_value': self.damageability_aga_negative,
            'equivalent_positive_value': self.equivalent_full_positive,
            'equivalent_negative_value': self.equivalent_full_negative,
            'equivalent_aga_positive_value': self.equivalent_aga_positive,
            'equivalent_aga_negative_value': self.equivalent_aga_negative,
        }

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def damageability_density(
        self,
        time_array:Optional[NDArray]=None,
        sample_rate:Annotated[int, Gt(0)]=50,
    ) -> tuple[NDArray, NDArray]:
        """
        Выполняет вычисление плотности повреждаемости.

        Args:
            time_array : ndarray, optional
                Массив значений времени. Если не задано - будет установлен стандартный
                массив значений в соотвествии с заданной частотой дискретизации процесса sample_rate.
                Default to None
            sample_rate : int
                Частота дискретизации процесса в герцах.
                Используется для подготови массива времени time_array, если 
                последний не задан.
                Deafult to 50
        
        Returns:
            damageability_density_positive : ndarray
                Массив значений плотности повреждаемости прямой реализации обработки на повреждаемость.
            damageability_density_negative : ndarray
                Массив значений плотности повреждаемости обратной реализации обработки на повреждаемость.
        
        """

        if time_array is None:
            # Инициализация стандартного массива времени
            time_array = self.__init_time_array(sample_rate=sample_rate,)
        else: 
            # Проверка размера входного массива
            if self._values.shape[0] != time_array.shape[0]:
                raise ValueError('The shape of «values» array must be same as «time_values» array.')


        # Инициализация массивов плотности повреждаемости
        # прямой реализации обработки на повреждаемость
        damageability_density_positive = _damageability_density(
            damageability_array=self._damageability_array_positive,
            time_array=time_array,
            start_indexes=self._start_cycle_indexes,
            end_indexes=self._end_cycle_indexes,
        )
        # обратной реализации обработки на повреждаемость
        damageability_density_negative = _damageability_density(
            damageability_array=self._damageability_array_negative,
            time_array=time_array,
            start_indexes=self._start_cycle_indexes,
            end_indexes=self._end_cycle_indexes,
        )

        return damageability_density_positive, damageability_density_negative


    @validate_call(config=dict(arbitrary_types_allowed=True))
    def damageability_density_plot(
        self,
        time_array:Optional[NDArray]=None,
        sample_rate:Annotated[int, Gt(0)]=50,
        dpi:Annotated[float, Gt(0.0)]=100,
        figsize_values:tuple[Annotated[float, Gt(0.0)],Annotated[float, Gt(0.0)]]=(16.0, 6.0),
        font_size_figure_title:Annotated[int, Gt(0)]=12,
        font_size_figure_labels:Annotated[int, Gt(0)]=10,
        font_size_figure_ticks:Annotated[int, Gt(0)]=10,
        damageability_density_positive_name:str='Плотность повреждаемости прямой реализации обработки на повреждаемость',
        damageability_density_negative_name:str='Плотность повреждаемости обратной реализации обработки на повреждаемость',
    ) -> Figure:
        """
        Выполняет вычисление плотности повреждаемости.

        Args:
            time_array : ndarray, optional
                Массив значений времени. Если не задано - будет установлен стандартный
                массив значений в соотвествии с заданной частотой дискретизации процесса sample_rate.
                Default to None
            sample_rate : int
                Частота дискретизации процесса в герцах.
                Используется для подготови массива времени time_array, если 
                последний не задан.
                Deafult to 50
            dpi : float
                Плотность пикселей изображеня (количество точек на дюйм).
                Default to 100.0
            figsize_values : tuple of float
                Figure width and height.
                Default to (16.0, 6.0)
            font_size_figure_title : int
                Размер шрифта наименования графика
                Default to 12
            font_size_figure_labels : int
                Размер шрифта подписей и легенды
                Default to 10
            font_size_figure_ticks : int
                Размер шрифта тиков
                Default to 10
            damageability_density_positive_name : str
                Подпись параметра значений плотности повреждаемости прямой реализации обработки на повреждаемость. 
                Default to 'Плотность повреждаемости прямой реализации обработки на повреждаемость'
            damageability_density_negative_name : str
                Подпись параметра значений плотности повреждаемости обратной реализации обработки на повреждаемость. 
                Default to 'Плотность повреждаемости обратной реализации обработки на повреждаемость'
        
        Returns:
            damageability_density_plot : matplotlib figure
                График плотности повреждаемости.
        
        """

        if time_array is None:
            # Инициализация стандартного массива времени
            time_array = self.__init_time_array(sample_rate=sample_rate,)
        else: 
            # Проверка размера входного массива
            if self._values.shape[0] != time_array.shape[0]:
                raise ValueError('The shape of «values» array must be same as «time_values» array.')


        # Инициализация массивов плотности повреждаемости
        # прямой реализации обработки на повреждаемость
        damageability_density_positive = _damageability_density(
            damageability_array=self._damageability_array_positive,
            time_array=time_array,
            start_indexes=self._start_cycle_indexes,
            end_indexes=self._end_cycle_indexes,
        )
        # обратной реализации обработки на повреждаемость
        damageability_density_negative = _damageability_density(
            damageability_array=self._damageability_array_negative,
            time_array=time_array,
            start_indexes=self._start_cycle_indexes,
            end_indexes=self._end_cycle_indexes,
        )

        return _damageability_density_plot(
            damageability_density_positive=damageability_density_positive,
            damageability_density_negative=damageability_density_negative,
            time_array=time_array,
            dpi=dpi,
            figsize_values=figsize_values,
            font_size_figure_title=font_size_figure_title,
            font_size_figure_labels=font_size_figure_labels,
            font_size_figure_ticks=font_size_figure_ticks,
            damageability_density_positive_name=damageability_density_positive_name,
            damageability_density_negative_name=damageability_density_negative_name,
        )


    def __init_time_array(self, sample_rate:int, ) -> NDArray:
        return np.multiply(np.arange(stop=self._values.shape[0], dtype=np.float64,), sample_rate,)


    def __add__(self, other):

        if isinstance(other, Damageability):

            source = self._init_data
            addendum = other._init_data
            
            if source.get('_fcd') != addendum.get('_fcd'):
                raise ValueError('Addition of damageability with different «fcd» is not allowed.')

            if source.get('_smoothing_value') != addendum.get('_smoothing_value'):
                warnings.warn('Addition of smoothing values is not possible. The average value will be output.')


            # Последовательное объединение массивов значений
            _values = np.hstack((
                source.get('_values'), 
                addendum.get('_values')
            ))
            # Среднее значение величины сглаживания
            _smoothing_value = np.mean((
                source.get('_smoothing_value'), 
                addendum.get('_smoothing_value')
            ))
            # Получение значения fcd
            _fcd = source.get('_fcd')
            # Последовательное объединение массивов экстремумов с переиндексацией
            _extremum_indexes = np.hstack((
                source.get('_extremum_indexes'), 
                np.add(
                    addendum.get('_extremum_indexes'), 
                    source.get('_values').size
                )
            ))
            # Последовательное объединение массивов индексов полных циклов с переиндексацией
            _start_cycle_indexes = np.hstack((
                source.get('_start_cycle_indexes'), 
                np.add(
                    addendum.get('_start_cycle_indexes'), 
                    source.get('_values').size
                )
            ))
            _end_cycle_indexes = np.hstack((
                source.get('_end_cycle_indexes'), 
                np.add(
                    addendum.get('_end_cycle_indexes'), 
                    source.get('_values').size
                )
            ))
            # Последовательное объединение массивов значений отнулевых циклов
            _zero_cycles_positive = np.hstack((
                source.get('_zero_cycles_positive'), 
                addendum.get('_zero_cycles_positive')
            ))
            _zero_cycles_negative = np.hstack((
                source.get('_zero_cycles_negative'), 
                addendum.get('_zero_cycles_negative')
            ))       
            # Последовательное объединение массивов значений повреждаемости
            _damageability_array_positive = np.hstack((
                source.get('_damageability_array_positive'), 
                addendum.get('_damageability_array_positive')
            ))
            _damageability_array_negative = np.hstack((
                source.get('_damageability_array_negative'), 
                addendum.get('_damageability_array_negative')
            ))

            # Амплитуды выделенных циклов
            _amplitudes = np.divide(np.abs(np.subtract(
                _values[_end_cycle_indexes], 
                _values[_start_cycle_indexes],
            )), 2)
            # Индексы сортировки циклов по возрастанию амплитуды
            _sort_indexes = np.argsort(_amplitudes)
            
            # Сортировка по возрастанию амплитуды
            _start_cycle_indexes = _start_cycle_indexes[_sort_indexes]
            _end_cycle_indexes = _end_cycle_indexes[_sort_indexes]
            _zero_cycles_positive = _zero_cycles_positive[_sort_indexes]
            _zero_cycles_negative = _zero_cycles_negative[_sort_indexes]
            _damageability_array_positive = _damageability_array_positive[_sort_indexes]
            _damageability_array_negative = _damageability_array_negative[_sort_indexes]

            # Словарь параметров сложенных параметров повреждаемости
            _damageability = dict(
                _values=_values,
                _smoothing_value=_smoothing_value,
                _fcd=_fcd,
                _extremum_indexes=_extremum_indexes,
                _start_cycle_indexes=_start_cycle_indexes,
                _end_cycle_indexes=_end_cycle_indexes,
                _zero_cycles_positive=_zero_cycles_positive,
                _zero_cycles_negative=_zero_cycles_negative,
                _damageability_array_positive=_damageability_array_positive,
                _damageability_array_negative=_damageability_array_negative,
            )

            return Damageability(**_damageability)

        else:
            return NotImplemented


    def __radd__(self, other):
        return self.__add__(other)


