use numpy::ndarray::{ArrayD, ArrayView, ArrayViewD, ArrayViewMutD, Dimension, Array1, Array2, ArrayView2};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use std::ops::Index;
use std::collections::{BTreeSet, HashMap};
use itertools::Itertools;
use rayon::prelude::*;

struct GLevel {
    level_id: usize,
    cache: Vec<HashMap<BTreeSet<i32>, (i32, i32)>>
}

#[pymodule]
fn tsp(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    fn do_infinity_add(a: i32, b: i32) -> i32 {
        if a == -1 || b == -1 {
            return -1;
        }
        return a + b;
    }

    fn do_infinity_gt(a: i32, b: i32) -> bool {
        if a == -1 {
            return b != -1 ;
        } else if b == -1 {
            return false;
        }
        return a > b;
    }

    // Solve the TSP using Held-Karp, returning the indices of the route.
    fn solve_problem(cost_matrix_py: ArrayViewD<'_, i64>, n: i32) -> ArrayD<i32> {
        let cost_slice = cost_matrix_py.to_slice().unwrap();
        let cost_slice_reduced_bits = cost_slice.iter().map(|elem| *elem as i32).collect::<Vec<i32>>();
        let cost_matrix = ArrayView::from_shape((n as usize, n as usize), &cost_slice_reduced_bits).unwrap();
        let mut levels = Vec::new();
        let mut level0 = GLevel {
            level_id: 0,
            cache: Vec::new()
        };
        for city_idx in 0..n {
            level0.cache.push(HashMap::new());
            level0.cache[city_idx as usize].insert(BTreeSet::new(), (cost_matrix[(0 as usize, city_idx as usize)], -1));
        }
        levels.push(level0);
        while levels.len() < (n-1) as usize {
            let mut level = GLevel {
                level_id: levels.len(),
                cache: Vec::new()
            };
            for _ in 0..n {
                level.cache.push(HashMap::new());
            }
            for cacheable_vec in (1..n).combinations(level.level_id).into_iter().par_bridge().map(|combo| {
                let mut temp_cache: Vec<(usize, BTreeSet<i32>, (i32, i32))> = Vec::with_capacity((n-1) as usize);
                for new_city_idx in 1..n {
                    let mut min_result: Option<(i32, i32)> = None;
                    let mut proceed = true;
                    for elem in &combo {
                        if *elem == new_city_idx {
                            proceed = false;
                            break;
                        }
                    }
                    if !proceed {
                        continue;
                    }
                    for city_idx in &combo {
                        if new_city_idx == *city_idx {
                            continue;
                        }
                        let mut comboset = BTreeSet::new();
                        for elem in &combo {
                            if elem != city_idx {
                                comboset.insert(*elem);
                            }
                        }
                        let result = (do_infinity_add(cost_matrix[(*city_idx as usize, new_city_idx as usize)], levels[level.level_id - 1].cache[*city_idx as usize].get(&comboset).unwrap().0), city_idx.clone());
                        if min_result.is_none() || do_infinity_gt(min_result.unwrap().0, result.0) {
                            min_result = Some(result);
                        }
                    }
                    if min_result.is_none() {
                        continue;
                    }
                    let mut comboset = BTreeSet::new();
                    for city_idx in &combo {
                        comboset.insert(*city_idx);
                    }
                    temp_cache.push((new_city_idx as usize, comboset, min_result.unwrap()));
                }
                temp_cache
            }).collect::<Vec<Vec<(usize, BTreeSet<i32>, (i32, i32))>>>() {
                for cacheable in cacheable_vec {
                    level.cache[cacheable.0].insert(cacheable.1, cacheable.2);
                }
            }
            levels.push(level);
        }
        let mut final_result: Option<(i32, i32)> = None;
        for city_idx in 1..n {
            let mut comboset = BTreeSet::new();
            for elem in 1..n {
                if elem != city_idx {
                    comboset.insert(elem);
                }
            }
            let result = (do_infinity_add(cost_matrix[(city_idx as usize, 0)], levels[(n-2) as usize].cache[city_idx as usize].get(&comboset).unwrap().0), city_idx.clone());
            if final_result.is_none() || do_infinity_gt(final_result.unwrap().0, result.0) {
                final_result = Some(result);
            }
        }
        let mut route = Vec::from([0]);
        let mut finals_tracker = final_result.clone().unwrap();
        let mut remaining = BTreeSet::new();
        for val in 1..n {
            remaining.insert(val);
        }
        while finals_tracker.1 != -1 {
            route.push(finals_tracker.1);
            remaining.remove(&finals_tracker.1);
            finals_tracker = *levels[remaining.len()].cache[finals_tracker.1 as usize].get(&remaining).unwrap();
        }
        route.push(0);
        route.reverse();
        Array1::from(route).into_dyn()
    }

    // wrapper of `solve_problem`
    #[pyfn(m)]
    #[pyo3(name = "solve_problem")]
    fn solve_problem_py<'py>(
        py: Python<'py>,
        cost_matrix_py: PyReadonlyArrayDyn<i64>,
        n: i64
    ) -> &'py PyArrayDyn<i32> {
        let cost_matrix_pyarr = cost_matrix_py.as_array();
        let z = solve_problem(cost_matrix_pyarr, n as i32);
        z.into_pyarray(py)
    }

    Ok(())
}

