use pyo3::prelude::*;
use tokio::runtime::Runtime;
use serde_json;
use tokio::signal;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use crate::config::{load_config};
use crate::world::World;
use crate::websocket::websocket;
use tokio::task::JoinHandle;


#[pyclass]
pub struct Simulation {
    world: World,
    bcast_tx: broadcast::Sender<String>,
    _server_handle: JoinHandle<()>,
    cancel_token: CancellationToken,
    rt: Runtime,
}

#[pymethods]
impl Simulation {
    #[new]
    pub fn new(config_path: String) -> PyResult<Self> {
        let rt = Runtime::new()?;
        let (world, bcast_tx, _server_handle, cancel_token) = rt.block_on(async {
            let config = load_config(&config_path).map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to load config: {}", e))
            })?;

            if !config.headless {
                println!("Starting simulation with UI server...");
            }

            let world = World::new(config);

            let (bcast_tx, _) = broadcast::channel::<String>(32);
            let cancel_token = CancellationToken::new();

            let server_handle = {
                let ws_filter = websocket(bcast_tx.clone());
                let cancel_clone = cancel_token.clone();
                tokio::spawn(async move {
                    warp::serve(ws_filter)
                    .bind(([127, 0, 0, 1], 3000)).await
                    .graceful(async move {
                        signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
                        cancel_clone.cancel();
                    })
                    .run().await;
                })
            };

            PyResult::Ok((world, bcast_tx, server_handle, cancel_token))
        })?;

        Ok(Simulation {
            world,
            bcast_tx,
            _server_handle,
            cancel_token,
            rt,
        })
    }

    pub fn step(&mut self) -> PyResult<()> {

        let world_state = self.world.get_state();
        let json_state = serde_json::to_string(&world_state)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("failed to serialize world state: {}", e)))?;

        if let Err(e) = self.bcast_tx.send(json_state) {
            println!("[DEBUG] Broadcast error: {}", e);
        }
        Ok(())
    }

    pub fn shutdown(&self) -> PyResult<()> {
        println!("Shutting down simulation server...");
        self.cancel_token.cancel();
        Ok(())
    }
}