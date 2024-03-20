import numpy as np
from matplotlib import pyplot
from matplotlib.collections import LineCollection

from ._allocate_full_cycles import _allocate_full_cycles

from numpy.typing import NDArray
from matplotlib.figure import Figure
from typing import Optional, Literal, Generator
from annotated_types import Annotated, Gt

from pydantic import validate_call


@validate_call(config=dict(arbitrary_types_allowed=True))
def visualization_full_cycles(
    values:NDArray,
    return_original_plot:bool=True,
    return_interim_main_plot:bool=True,
    return_interim_remainder_plot:bool=True,
    return_result_plot:bool=True,
    set_legend:bool=False,
    set_extremum_points:Optional[Literal['indexes','values']]=None,
    sort:bool=False,
    dpi:Annotated[float, Gt(0.0)]=100,
    figsize_values:tuple[Annotated[float, Gt(0.0)],Annotated[float, Gt(0.0)]]=(6.496, 2.436),
    font_size_figure_title:Annotated[int, Gt(0)]=12,
    font_size_figure_labels:Annotated[int, Gt(0)]=10,
    font_size_figure_ticks:Annotated[int, Gt(0)]=10,    
) -> list[Figure]:
    """
    Визуализирует процесс выделения полных циклов 

    Args:
        values : ndarray
            Массив обрабатываемых значений. Значения должны быть массивом экстремумов.
        return_original_plot : bool
            Возвращать график исходного состояния экстремумов.
            Default to True
        return_interim_main_plot : bool
            Возвращать массив графиков основного процесса выделения полных циклов.
            Default to True
        return_interim_remainder_plot : bool
            Возвращать график процесса выделения полных циклов из массива остатков.
            Default to True
        return_result_plot : bool
            Возвращать график массива выделенных полных циклов.
            Default to True
        set_legend : bool
            Устанавливать легенду на графиках.
            Default to False
        set_extremum_points : 'indexes' or'values', optional
            Устанавливать подписи индексов/значений точек на графиках.
            Default to None
        sort : bool
            Выполнять сортировку выделенных полных циклов по возрастанию амплитуды
            Default to False
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

    Returns:
        list of matplotlib figure
    
    """

    return _visualization_full_cycles(
        values=values,
        return_original_plot=return_original_plot,
        return_interim_main_plot=return_interim_main_plot,
        return_interim_remainder_plot=return_interim_remainder_plot,
        return_result_plot=return_result_plot,        
        set_legend=set_legend,
        set_extremum_points=set_extremum_points,
        sort=sort,
        dpi=dpi,
        figsize_values=figsize_values,
        font_size_figure_title=font_size_figure_title,
        font_size_figure_labels=font_size_figure_labels,
        font_size_figure_ticks=font_size_figure_ticks,
    )


def _visualization_full_cycles(
    values:NDArray,
    return_original_plot:bool,
    return_interim_main_plot:bool,
    return_interim_remainder_plot:bool,
    return_result_plot:bool,
    set_legend:bool,
    set_extremum_points:Optional[Literal['indexes','values']],
    sort:bool,
    dpi:Annotated[float, Gt(0.0)],
    figsize_values:tuple[Annotated[float, Gt(0.0)],Annotated[float, Gt(0.0)]],
    font_size_figure_title:Annotated[int, Gt(0)],
    font_size_figure_labels:Annotated[int, Gt(0)],
    font_size_figure_ticks:Annotated[int, Gt(0)],
    **kwargs,
) -> list[Figure]:
    
    # Получение индексов полных циклов
    (
        (start_full_cycle_main, end_full_cycle_main), 
        (start_full_cycle_remainder, end_full_cycle_remainder,), 
        (start_full_cycle_last, end_full_cycle_last),
    ) = _allocate_full_cycles(
        values=values, 
        result_type='indexes',
        sort=False,
        fcd=None,
        raw_data=True,
    )

    # Инициализация списка фигур
    figure_list = list()

    
    if return_original_plot:

        # Массив координат
        indexes = np.arange(stop=values.size)

        # Исходное состояние данных
        figure = __original_plot(
            x_values=indexes,
            y_values=values,
            set_extremum_points=set_extremum_points,
            set_legend=set_legend,
            dpi=dpi,
            figsize_values=figsize_values,
            font_size_figure_title=font_size_figure_title,
            font_size_figure_labels=font_size_figure_labels,
            font_size_figure_ticks=font_size_figure_ticks,
        )
        figure_list.append(figure)
    

    if return_interim_main_plot:

        # Получение генератора для визаулизации выделения полных циклов
        iterator = __full_cycle_main_iterations(
            values=values,
            start_full_cycle=start_full_cycle_main,
            end_full_cycle=end_full_cycle_main,
        )

        try:
            while True:

                # Получение словаря данных для построения
                iteration_dict = next(iterator)
                # Выделение единичного цикла
                figure = __cycle_main_plot(
                    **iteration_dict,
                    figsize_values=figsize_values,
                    set_legend=set_legend,
                    set_extremum_points=set_extremum_points,
                    dpi=dpi,
                    font_size_figure_title=font_size_figure_title,
                    font_size_figure_labels=font_size_figure_labels,
                    font_size_figure_ticks=font_size_figure_ticks,
                )
                figure_list.append(figure)

        except StopIteration:
            ...
        finally:
            del iterator
        
    
    if return_interim_remainder_plot:

        # Получение словаря данных для построения
        iteration_dict = __full_cycle_remainder_iteration(
            values=values,
            start_full_cycle_main=start_full_cycle_main,
            end_full_cycle_main=end_full_cycle_main,
            start_full_cycle_remainder=start_full_cycle_remainder,
            end_full_cycle_remainder=end_full_cycle_remainder,
        )

        # Визуализация выделения полных циклов из массива остатков
        figure = __cycle_remainder_plot(
            **iteration_dict,
            y_values=values,
            figsize_values=figsize_values,
            set_legend=set_legend,
            set_extremum_points=set_extremum_points,
            dpi=dpi,
            font_size_figure_title=font_size_figure_title,
            font_size_figure_labels=font_size_figure_labels,
            font_size_figure_ticks=font_size_figure_ticks,
        )
        figure_list.append(figure)


    if return_result_plot:
        
        # Объединение массивов индексов выделенных циклов
        start_full_cycle = np.concatenate(( start_full_cycle_main, start_full_cycle_remainder, start_full_cycle_last, ))
        end_full_cycle = np.concatenate(( end_full_cycle_main, end_full_cycle_remainder, end_full_cycle_last ))

        if sort:

            # Амплитуды выделенных циклов
            amplitudes = np.divide(np.abs(np.subtract(
                values[end_full_cycle], 
                values[start_full_cycle],
            )), 2)
            
            # Индексы сортировки циклов по возрастанию амплитуды
            sort_indexes = np.argsort(amplitudes)

            # Сортировка циклов по возрастанию амплитуды
            start_full_cycle = start_full_cycle[sort_indexes]
            end_full_cycle = end_full_cycle[sort_indexes]

        # Визуализация выделенных циклов
        figure = __result_plot(
            y_values=values,
            start_full_cycle=start_full_cycle,
            end_full_cycle=end_full_cycle,
            set_extremum_points=set_extremum_points,
            set_legend=set_legend,
            dpi=dpi,
            figsize_values=figsize_values,
            font_size_figure_title=font_size_figure_title,
            font_size_figure_labels=font_size_figure_labels,
            font_size_figure_ticks=font_size_figure_ticks,
        )

        figure_list.append(figure)
        
    
    return figure_list


def __cycle_main_plot(
    x_before:NDArray,
    y_before:NDArray,
    x_after:NDArray,
    y_after:NDArray,
    x_left:tuple[int, int],
    y_left:tuple[float, float],
    x_right:tuple[int, int],
    y_right:tuple[float, float],
    x_selected:tuple[int, int],
    y_selected:tuple[float, float],
    x_new:tuple[int, int],
    y_new:tuple[float, float],
    figsize_values:tuple[Annotated[float, Gt(0.0)],Annotated[float, Gt(0.0)]],
    set_legend:bool=False,
    set_extremum_points:Optional[Literal['indexes','values']]=None,
    dpi:Annotated[float, Gt(0.0)]=100,
    font_size_figure_title:Annotated[int, Gt(0)]=12,
    font_size_figure_labels:Annotated[int, Gt(0)]=10,
    font_size_figure_ticks:Annotated[int, Gt(0)]=10,
    **kwargs,
) -> Figure:
    
    # Turn the interactive mode off
    pyplot.ioff()
    
    figure, axes = pyplot.subplots(
        nrows=1, ncols=1, 
        figsize=figsize_values, 
        facecolor='white', 
        dpi=dpi,
        squeeze=True,
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,)

    axes.plot(x_before, y_before, color='red', linestyle='solid', linewidth=1.0, )
    axes.plot(x_after, y_after, color='red', linewidth=1.0, )

    __label_right = 'removed' if x_left is None else None
    axes.plot(x_left, y_left, color='red', linestyle='dashed', linewidth=1.0, label='removed', )
    axes.plot(x_right, y_right, color='red', linestyle='dashed', linewidth=1.0, label=__label_right, )
    
    axes.plot(x_selected, y_selected, color='green', linestyle='dashed', linewidth=1.0, label='cycle', )
    
    axes.plot(x_new, y_new, color='blue', linestyle='solid', linewidth=1.0, label='new', )
    
    # Установка подписей точек
    if set_extremum_points is not None:

        x_text_values = np.concatenate((x_before, x_selected, x_after))
        y_text_values = np.concatenate((y_before, y_selected, y_after))
        s_text_dict = dict(indexes = x_text_values, values = y_text_values,)
        s_text_values = s_text_dict.get(set_extremum_points)

        for x_text, y_text, s_text in zip(x_text_values, y_text_values, s_text_values,):
            axes.text(
                x=x_text, y=y_text, s=s_text,
                horizontalalignment='center',
                verticalalignment='center', 
                fontsize=font_size_figure_labels,
            )
    

    if set_legend:
        
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
            ncols=3,
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

    return figure


def __cycle_remainder_plot(
    y_values:NDArray,
    indexes:NDArray,
    start_removed_indexes:NDArray,
    end_removed_indexes:NDArray,
    start_selected_indexes:NDArray,
    end_selected_indexes:NDArray,
    figsize_values:tuple[Annotated[float, Gt(0.0)],Annotated[float, Gt(0.0)]],
    set_legend:bool=False,
    set_extremum_points:Optional[Literal['indexes','values']]=None,
    dpi:Annotated[float, Gt(0.0)]=100,
    font_size_figure_title:Annotated[int, Gt(0)]=12,
    font_size_figure_labels:Annotated[int, Gt(0)]=10,
    font_size_figure_ticks:Annotated[int, Gt(0)]=10,
    **kwargs,
) -> Figure:
    
    # Turn the interactive mode off
    pyplot.ioff()
    
    figure, axes = pyplot.subplots(
        nrows=1, ncols=1, 
        figsize=figsize_values, 
        facecolor='white', 
        dpi=dpi,
        squeeze=True,
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,)

    removed_cycles = [
        ((y_start, y_values[y_start]), (y_stop, y_values[y_stop])) 
        for y_start, y_stop
        in zip(start_removed_indexes, end_removed_indexes,)
    ]
    removed_line_collection = LineCollection(
        segments=removed_cycles, 
        linewidths=1.0,
        linestyle='dashed',
        color='red', 
        label='removed',
    )

    selected_cycles = [
        ((y_start, y_values[y_start]), (y_stop, y_values[y_stop])) 
        for y_start, y_stop
        in zip(start_selected_indexes, end_selected_indexes,)
    ]
    selected_line_collection = LineCollection(
        segments=selected_cycles, 
        linewidths=1.0,
        linestyle='solid',
        color='green',
        label='cycle',
    )

    axes.add_collection(removed_line_collection)
    axes.add_collection(selected_line_collection)
    axes.autoscale()

    # Установка подписей точек
    if set_extremum_points is not None:

        x_text_values = indexes
        y_text_values = y_values[indexes]
        s_text_dict = dict(indexes = x_text_values, values = y_text_values,)
        s_text_values = s_text_dict.get(set_extremum_points)

        for x_text, y_text, s_text in zip(x_text_values, y_text_values, s_text_values,):
            axes.text(
                x=x_text, y=y_text, s=s_text,
                horizontalalignment='center',
                verticalalignment='center', 
                fontsize=font_size_figure_labels,
            )

    if set_legend:
        
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
            ncols=3,
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

    return figure


def __original_plot(
    x_values:NDArray,
    y_values:NDArray,
    set_extremum_points:Optional[Literal['indexes','values']],
    set_legend:bool,
    dpi:Annotated[float, Gt(0.0)],
    figsize_values:tuple[Annotated[float, Gt(0.0)],Annotated[float, Gt(0.0)]],
    font_size_figure_title:Annotated[int, Gt(0)],
    font_size_figure_labels:Annotated[int, Gt(0)],
    font_size_figure_ticks:Annotated[int, Gt(0)],
    **kwargs,
) -> Figure:
    
    # Turn the interactive mode off
    pyplot.ioff()
    
    figure, axes = pyplot.subplots(
        nrows=1, ncols=1, 
        figsize=figsize_values, 
        facecolor='white', 
        dpi=dpi,
        squeeze=True,
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,)

    axes.plot(x_values, y_values, color='red', linestyle='solid', linewidth=1.0, )
    
    # Установка подписей точек
    if set_extremum_points is not None:

        s_text_dict = dict(indexes = x_values, values = y_values,)
        s_text_values = s_text_dict.get(set_extremum_points)

        for x_text, y_text, s_text in zip(x_values, y_values, s_text_values,):
            axes.text(
                x=x_text, y=y_text, s=s_text,
                horizontalalignment='center',
                verticalalignment='center', 
                fontsize=font_size_figure_labels,
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
    
    return figure


def __result_plot(
    y_values:NDArray,
    start_full_cycle:NDArray,
    end_full_cycle:NDArray,
    set_extremum_points:Optional[Literal['indexes','values']],
    set_legend:bool,
    dpi:Annotated[float, Gt(0.0)],
    figsize_values:tuple[Annotated[float, Gt(0.0)],Annotated[float, Gt(0.0)]],
    font_size_figure_title:Annotated[int, Gt(0)],
    font_size_figure_labels:Annotated[int, Gt(0)],
    font_size_figure_ticks:Annotated[int, Gt(0)],
    **kwargs,
) -> Figure:
    
    # Turn the interactive mode off
    pyplot.ioff()
    
    figure, axes = pyplot.subplots(
        nrows=1, ncols=1, 
        figsize=figsize_values, 
        facecolor='white', 
        dpi=dpi,
        squeeze=True,
    )
    figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0,)

    cycles = [
        ((i, y_start), (i, y_stop)) 
        for i, (y_start, y_stop)
        in enumerate(zip(y_values[start_full_cycle], y_values[end_full_cycle],))
    ]

    line_collection = LineCollection(
        segments=cycles, 
        linewidths=2,
        linestyle='solid',
    )

    axes.add_collection(line_collection)
    axes.autoscale()
    # axes.margins(0.1)
    
    # Установка подписей точек
    if set_extremum_points is not None:
        
        x_text_values = np.concatenate((
            np.arange(stop=start_full_cycle.size,), 
            np.arange(stop=end_full_cycle.size,), 
        ))
        y_text_values = np.concatenate((
            y_values[start_full_cycle], 
            y_values[end_full_cycle], 
        ))
        s_text_dict = dict(
            indexes = np.concatenate((start_full_cycle, end_full_cycle, )), 
            values = y_values[np.concatenate((start_full_cycle, end_full_cycle, ))],
        )
        s_text_values = s_text_dict.get(set_extremum_points)

        for x_text, y_text, s_text in zip(x_text_values, y_text_values, s_text_values,):
            axes.text(
                x=x_text, y=y_text, s=s_text,
                horizontalalignment='center',
                verticalalignment='center', 
                fontsize=font_size_figure_labels,
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
    
    return figure


def __full_cycle_main_iterations(
    values:NDArray,
    start_full_cycle:NDArray,
    end_full_cycle:NDArray,
) -> Generator[dict[str,tuple|NDArray], None, None]:
    
    # Индексы крайних точек массива
    idx_first_point, idx_last_point = np.min(start_full_cycle), np.max(end_full_cycle)
    # Позиции крайних точек массива в индексах полных циклов
    position_first_point = np.where(np.in1d(ar1=start_full_cycle, ar2=idx_first_point,))[0].item()
    position_last_point = np.where(np.in1d(ar1=end_full_cycle, ar2=idx_last_point,))[0].item()
    # Перенос цикла, образованного крайними точками, если таковой имеется, в конец массива
    if position_first_point == position_last_point:
        start_full_cycle = np.append(
            np.delete(start_full_cycle, position_first_point), 
            start_full_cycle[position_first_point]
        )
        end_full_cycle = np.append(
            np.delete(end_full_cycle, position_last_point), 
            end_full_cycle[position_last_point]
        )

    x_values = np.arange(stop=values.size, dtype=np.int64, )
    y_values = values


    for idx_start_cycle, idx_end_cycle in zip(start_full_cycle, end_full_cycle):

        # Позиции точек отобранного цикла
        position_start_selected, position_end_selected = np.where(np.in1d(ar1=x_values, ar2=[idx_start_cycle, idx_end_cycle], ))[0]
        # Индексы и значения точек отобранного цикла
        x_selected = x_values[position_start_selected].item(), x_values[position_end_selected].item()
        y_selected = y_values[position_start_selected].item(), y_values[position_end_selected].item()

        # Позиции точек нового образованного цикла
        position_start_new = position_start_selected-1 if position_start_selected > 0 else position_start_selected
        position_end_new = position_end_selected+1 if position_end_selected < x_values.size-1 else position_end_selected
        # Индексы и значения точек нового образованного цикла
        x_new = x_values[position_start_new].item(), x_values[position_end_new].item()
        y_new = y_values[position_start_new].item(), y_values[position_end_new].item()

        # Позиции точек удаленного цикла слева
        position_start_left, position_end_left = position_start_new, position_start_selected
        # Индексы и значения точек удаленного цикла слева
        x_left = x_values[position_start_left].item(), x_values[position_end_left].item()
        y_left = y_values[position_start_left].item(), y_values[position_end_left].item()

        # Позиции точек удаленного цикла справа
        position_start_right, position_end_right = position_end_selected, position_end_new
        # Индексы и значения точек удаленного цикла справа
        x_right = x_values[position_start_right].item(), x_values[position_end_right].item()
        y_right = y_values[position_start_right].item(), y_values[position_end_right].item()

        # Неизменная часть массива
        x_before = x_values[:position_start_left+1]
        x_after = x_values[position_end_right:]
        y_before = y_values[:position_start_left+1]
        y_after = y_values[position_end_right:]

        # Обновление исходных массивов после выделения цикла
        x_values = np.concatenate(( x_before, x_after, ))
        y_values = np.concatenate(( y_before, y_after, ))

        iteration_dict = dict(
            x_before=x_before,
            x_after=x_after,
            y_before=y_before,
            y_after=y_after,
            x_selected=x_selected,
            y_selected=y_selected,
            x_new=x_new,
            y_new=y_new,
            x_left=x_left,
            y_left=y_left,
            x_right=x_right,
            y_right=y_right,
        )

        yield iteration_dict


def __full_cycle_remainder_iteration(
    values:NDArray,
    start_full_cycle_main:NDArray,
    end_full_cycle_main:NDArray,
    start_full_cycle_remainder:NDArray,
    end_full_cycle_remainder:NDArray,
):
    
    # Массив координат
    indexes = np.setdiff1d(
        ar1=np.arange(stop=values.shape[0]),
        ar2=np.concatenate((start_full_cycle_main,end_full_cycle_main)),
    )

    start_removed_indexes = np.setdiff1d(
        ar1=indexes[ :-1],
        ar2=start_full_cycle_remainder,
    )
    end_removed_indexes = np.setdiff1d(
        ar1=indexes[1:  ],
        ar2=end_full_cycle_remainder,
    )

    return dict(
        indexes=indexes,
        start_removed_indexes=start_removed_indexes,
        end_removed_indexes=end_removed_indexes,
        start_selected_indexes=start_full_cycle_remainder,
        end_selected_indexes=end_full_cycle_remainder,
    )
    
