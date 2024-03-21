import numpy as np

from damageability._allocate_full_cycles import allocate_full_cycles


extremum_values = np.array([
    0.079,  0.003,  0.138, -0.054, -0.036, -0.099,  0.207, -0.014, 0.016,  
    0.004,  0.07 ,  0.045,  0.199,  0.034,  0.14 ,  0.12 , 0.24 ,  0.084,
])

start_cycle_values = np.array([
   -0.054,  0.016,  0.14 ,  0.07 ,  0.199,  0.207,  0.003, -0.099, 0.079,
])
end_cycle_values = np.array([
    -0.036,  0.004,  0.12 ,  0.045,  0.034, -0.014,  0.138,  0.24 ,0.084,
])


def test_allocate_full_cycles_values():
    start_cycle_values_, end_cycle_values_ = allocate_full_cycles(
        values=extremum_values,
        result_type='values',
        sort=True,
        fcd=None,
    )

    amplitudes = np.divide(np.abs(np.subtract(end_cycle_values,start_cycle_values)), 2)
    sort_indexes = np.argsort(amplitudes)

    print(start_cycle_values_, end_cycle_values_)
    assert (
        np.all(np.equal(start_cycle_values_, start_cycle_values[sort_indexes])) 
        and 
        np.all(np.equal(end_cycle_values_, end_cycle_values[sort_indexes]))
    )



