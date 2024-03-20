import numpy as np
from matplotlib import pyplot
from functools import reduce

from numpy.typing import NDArray
from matplotlib.figure import Figure
from annotated_types import Annotated, Gt

from pydantic import validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def damageability_density_plot(
    damageability_density_positive:NDArray,
    damageability_density_negative:NDArray,
    time_array:NDArray,
    dpi:Annotated[float, Gt(0.0)]=100,
    figsize_values:tuple[Annotated[float, Gt(0.0)],Annotated[float, Gt(0.0)]]=(6.496, 2.436),
    font_size_figure_title:Annotated[int, Gt(0)]=12,
    font_size_figure_labels:Annotated[int, Gt(0)]=10,
    font_size_figure_ticks:Annotated[int, Gt(0)]=10,
    damageability_density_positive_name:str='Плотность повреждаемости прямой реализации обработки на повреждаемость',
    damageability_density_negative_name:str='Плотность повреждаемости обратной реализации обработки на повреждаемость',
) -> Figure:
    """
    Выполняет построение графика плотности повреждаемости.


    Args:
        damageability_density_positive : ndarray
            Массив значений плотности повреждаемости прямой реализации обработки на повреждаемость.
        damageability_density_negative : ndarray
            Массив значений плотности повреждаемости обратной реализации обработки на повреждаемость.
        time_array : ndarray
            Массив значений времени.
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

def _damageability_density_plot(
    damageability_density_positive:NDArray,
    damageability_density_negative:NDArray,
    time_array:NDArray,
    dpi:Annotated[float, Gt(0.0)],
    figsize_values:tuple[Annotated[float, Gt(0.0)],Annotated[float, Gt(0.0)]],
    font_size_figure_title:Annotated[int, Gt(0)],
    font_size_figure_labels:Annotated[int, Gt(0)],
    font_size_figure_ticks:Annotated[int, Gt(0)],
    damageability_density_positive_name:str,
    damageability_density_negative_name:str,
) -> Figure:

    # Turn the interactive mode off
    pyplot.ioff()

    # Создание фигуры и осей
    figure, axes = pyplot.subplots(
        nrows=1, ncols=1, 
        figsize=figsize_values, 
        facecolor='white', 
        dpi=dpi,
        squeeze=True,
    )
    # Размеры полей
    figure.subplots_adjust(
        left=0.0,
        right=1.0,
        bottom=0.0,
        top=1.0,
    )
    
    # Инициализация удвоенных массивов данных (точка начала, точка конца)
    time_array_double = np.repeat(a=time_array, repeats=2,)
    ddp_double = reduce(np.append, [0, np.repeat(a=damageability_density_positive, repeats=2,), 0])
    ddn_double = reduce(np.append, [0, np.repeat(a=damageability_density_negative, repeats=2,), 0])

    # Построение графика
    axes.fill_between(
        x=time_array_double, 
        y1=ddp_double, 
        facecolor='red',
        edgecolor='red',
        linestyle='solid',
        linewidth=0.7,
        alpha=0.5, 
        label=damageability_density_positive_name,
    )
    axes.fill_between(
        x=time_array_double, 
        y1=np.negative(ddn_double), 
        facecolor='blue',
        edgecolor='blue',
        linestyle='solid',
        linewidth=0.7,
        alpha=0.5, 
        label=damageability_density_negative_name,
    )

    # Установка легенды
    axes.legend(
        fontsize=font_size_figure_labels,
        handlelength=2.0,
        handletextpad=0.8,
        columnspacing=2.0,
        loc='lower left',
        bbox_to_anchor=(0.0, 1.0, 0.999, 0.0,),
        mode='expand',
        frameon=True,
        fancybox=True, 
        shadow=True, 
        ncols=1,
        borderaxespad=0.3,
    )

    # Установка параметров сетки
    axes.minorticks_on()
    axes.grid(which='major', axis='both', color='gray', linewidth='0.5', linestyle='-')
    axes.grid(which='minor', axis='both', color='gray', linewidth='0.5', linestyle=':')

    # Установка размера шрифтов подписей осей
    axes.tick_params(axis='both', which='both', labelsize=font_size_figure_ticks, )
    # Установка размера шрифта коэффициента значений тиков
    axes.xaxis.offsetText.set_fontsize(fontsize=font_size_figure_ticks,)
    axes.yaxis.offsetText.set_fontsize(fontsize=font_size_figure_ticks,)
    # Установка положения коэффициента значений тиков
    axes.xaxis.offsetText.set_horizontalalignment(align='right',)
    axes.yaxis.offsetText.set_horizontalalignment(align='right',)
    
    # Закрытие графика
    pyplot.close() 
    
    return figure

