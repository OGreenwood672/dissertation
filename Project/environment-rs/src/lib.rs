use pyo3::prelude::*;

pub mod agent;
pub mod config;
pub mod grid_map;
pub mod sim;
pub mod location;
pub mod resource;
pub mod station;
pub mod world;
pub mod websocket;

use crate::sim::Simulation;

#[pyfunction]
pub fn print_test() {
    println!("Hello! This should print!");
}

#[pymodule]
fn environment(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Simulation>()?;
    m.add_function(wrap_pyfunction!(print_test, m)?)?;
    Ok(())
}
