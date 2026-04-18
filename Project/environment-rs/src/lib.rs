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

#[pymodule]
fn environment(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Simulation>()?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::PyModule;

    #[test]
    fn registers_simulation_class() {
        Python::initialize();

        Python::attach(|py| {
            let m = PyModule::new(py, "environment").unwrap();
            environment(py, &m).unwrap();

            assert!(m.getattr("Simulation").is_ok());
        });
    }
}