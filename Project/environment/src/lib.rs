use pyo3::prelude::*;

pub mod agent;
pub mod config;
pub mod sim;
pub mod location;
pub mod resource;
pub mod station;
pub mod world;
pub mod websocket;

use crate::sim::run_sim_blocking;

#[pyfunction]
pub fn print_test() {
    println!("Hello! This should print!");
}

#[pymodule]
fn environment(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_sim_blocking, m)?)?;
    m.add_function(wrap_pyfunction!(print_test, m)?)?;
    Ok(())
}
