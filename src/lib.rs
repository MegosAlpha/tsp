use numpy::ndarray::{ArrayD, ArrayView, ArrayViewD, Array1};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};
use std::collections::{BTreeSet, HashMap};
use itertools::Itertools;
use rayon::prelude::*;

// Minor Space Optimization: remove this struct, use just a vector of caches instead
// TODO: Determine if there is a better data structure for the cache
// To replace BTreeSet alone, we need a hashable set data structure for use with HashMap
// We can use BTreeMap instead, but then need an orderable set
struct GLevel {
    level_id: usize,
    cache: Vec<HashMap<BTreeSet<u8>, (i32, u8)>>
}

#[pymodule]
fn tsp(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Infinity is -1 in this implementation.
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

    /// Solve the TSP using Held-Karp, returning the indices of the completed route.
    fn solve_problem(cost_matrix_py: ArrayViewD<'_, i64>, n: u8) -> ArrayD<u8> {
        // Process the cost matrix from Python
        let cost_slice = cost_matrix_py.to_slice().unwrap();
        let cost_slice_reduced_bits = cost_slice.iter().map(|elem| *elem as i32).collect::<Vec<i32>>();
        let cost_matrix = ArrayView::from_shape((n as usize, n as usize), &cost_slice_reduced_bits).unwrap();
        // Establish cache / sort-of DP table
        let mut levels = Vec::new();
        // Manually populate first level from cost matrix
        let mut level0 = GLevel {
            level_id: 0,
            cache: Vec::new()
        };
        for city_idx in 1..n {
            level0.cache.push(HashMap::new());
            level0.cache[(city_idx-1) as usize].insert(BTreeSet::new(), (cost_matrix[(0 as usize, city_idx as usize)], 0));
        }
        levels.push(level0);
        // Calculate each additional level
        // TODO: Consider partial level construction to seed another algorithm / additional stage
        // At time of writing, the problem allocates too much memory on my PC around PS >=25.
        while levels.len() < (n-1) as usize {
            let mut level = GLevel {
                level_id: levels.len(),
                cache: Vec::new()
            };
            // Create a path to value map (indices implicitly map the cities themselves)
            for _ in 1..n {
                level.cache.push(HashMap::new());
            }
            // Parallelization step: each combination runs in the Rayon thread pool
            // We take every combination (subset) of size equal to level
            for cacheable_vec in (1..n).combinations(level.level_id).into_iter().par_bridge().map(|combo| {
                // The temporary cache represents the results of each combination
                let mut temp_cache: Vec<(usize, BTreeSet<u8>, (i32, u8))> = Vec::with_capacity((n-1) as usize);
                // For every city...
                for new_city_idx in 1..n {
                    // Select the minimum result using the partial path,
                    // which is represented by the combination.
                    let mut min_result: Option<(i32, u8)> = None;
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
                    // Test the subsets of the partial paths to minimize across.
                    for city_idx in &combo {
                        if new_city_idx == *city_idx {
                            continue;
                        }
                        // Construct the partial path subset
                        let mut comboset = BTreeSet::new();
                        for elem in &combo {
                            if elem != city_idx {
                                comboset.insert(*elem);
                            }
                        }
                        let result = (do_infinity_add(cost_matrix[(*city_idx as usize, new_city_idx as usize)], levels[level.level_id - 1].cache[(*city_idx-1) as usize].get(&comboset).unwrap_or(&(-1,0)).0), city_idx.clone());
                        if result.0 != -1 && (min_result.is_none() || do_infinity_gt(min_result.unwrap().0, result.0)) {
                            min_result = Some(result);
                        }
                    }
                    // If no (finite) results are found, add nothing to the cache
                    if min_result.is_none() {
                        continue;
                    }
                    let mut comboset = BTreeSet::new();
                    for city_idx in &combo {
                        comboset.insert(*city_idx);
                    }
                    // Cache the winning result in the temp_cache
                    temp_cache.push(((new_city_idx - 1) as usize, comboset, min_result.unwrap()));
                }
                temp_cache
            }).collect::<Vec<Vec<(usize, BTreeSet<u8>, (i32, u8))>>>() {
                // Combine all the temporary cache entries into the level cache
                for cacheable in cacheable_vec {
                    level.cache[cacheable.0].insert(cacheable.1, cacheable.2);
                }
            }
            // Complete the level, adding it to the main cache
            levels.push(level);
        }
        // Complete the final level separately
        let mut final_result: Option<(i32, u8)> = None;
        for city_idx in 1..n {
            let mut comboset = BTreeSet::new();
            for elem in 1..n {
                if elem != city_idx {
                    comboset.insert(elem);
                }
            }
            let result = (do_infinity_add(cost_matrix[(city_idx as usize, 0)], levels[(n-2) as usize].cache[(city_idx-1) as usize].get(&comboset).unwrap().0), city_idx.clone());
            if final_result.is_none() || do_infinity_gt(final_result.unwrap().0, result.0) {
                final_result = Some(result);
            }
        }
        // Perform back-tracking to construct the route
        let mut route = Vec::from([0]);
        let mut finals_tracker = final_result.clone().unwrap();
        let mut remaining = BTreeSet::new();
        for val in 1..n {
            remaining.insert(val);
        }
        loop {
            route.push(finals_tracker.1);
            remaining.remove(&finals_tracker.1);
            finals_tracker = *levels[remaining.len()].cache[(finals_tracker.1-1) as usize].get(&remaining).unwrap();
            if finals_tracker.1 == 0 {
                break;
            }
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
    ) -> &'py PyArrayDyn<u8> {
        let cost_matrix_pyarr = cost_matrix_py.as_array();
        let z = solve_problem(cost_matrix_pyarr, n as u8);
        z.into_pyarray(py)
    }

    Ok(())
}

