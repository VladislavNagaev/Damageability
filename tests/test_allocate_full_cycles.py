import numpy as np

from damageability._allocate_full_cycles import _allocate_full_cycles


extremum_values = np.array([
    0.079,  0.003,  0.138, -0.054, -0.036, -0.099,  0.207, -0.014, 0.016,  
    0.004,  0.07 ,  0.045,  0.199,  0.034,  0.14 ,  0.12 , 0.24 ,  0.084,
])

start_cycle_indexes = np.array([ 3,  8, 14, 10, 12,  6,  1,  5,  0])
end_cycle_indexes = np.array([ 4,  9, 15, 11, 13,  7,  2, 16, 17])

start_cycle_values = np.array([
   -0.054,  0.016,  0.14 ,  0.07 ,  0.199,  0.207,  0.003, -0.099, 0.079,
])
end_cycle_values = np.array([
    -0.036,  0.004,  0.12 ,  0.045,  0.034, -0.014,  0.138,  0.24 ,0.084,
])


def test_allocate_full_cycles_values():
    start_cycle_values_, end_cycle_values_ = _allocate_full_cycles(
        values=extremum_values,
        result_type='values',
        sort=False,
        fcd=None,
    )
    print(start_cycle_values_, end_cycle_values_)
    assert (
        np.all(np.equal(start_cycle_values_, start_cycle_values)) 
        and 
        np.all(np.equal(end_cycle_values_, end_cycle_values))
    )


def test_allocate_full_cycles_indexes():
    start_cycle_indexes_, end_cycle_indexes_ = _allocate_full_cycles(
        values=extremum_values,
        result_type='indexes',
        sort=False,
        fcd=None,
    )
    print(start_cycle_indexes_, end_cycle_indexes_)
    assert (
        np.all(np.equal(start_cycle_indexes_, start_cycle_indexes)) 
        and 
        np.all(np.equal(end_cycle_indexes_, end_cycle_indexes))
    )


