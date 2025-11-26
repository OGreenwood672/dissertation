use pyo3::prelude::*;
use tokio::runtime::Runtime;
use serde_json;
use tokio::signal;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use crate::agent::{Action, ToActions};
use crate::config::{load_config};
use crate::world::World;
use crate::websocket::websocket;
use tokio::task::JoinHandle;
use std::iter::repeat_with;


#[pyclass]
pub struct Simulation {
    worlds: Vec<World>,
    headless: bool,
    bcast_tx: broadcast::Sender<String>,
    _server_handle: JoinHandle<()>,
    cancel_token: CancellationToken,
    #[expect(unused)]
    rt: Runtime,
}

#[pymethods]
impl Simulation {
    #[new]
    pub fn new(config_path: String) -> PyResult<Self> {

        let rt = Runtime::new()?;

        let config = load_config(&config_path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Failed to load config: {}", e))
        })?;

        let is_headless = config.headless;
        if !is_headless {
            println!("Starting simulation with UI server...");
        } else {
            println!("Starting simulation without UI server...");
        }

        let (worlds, bcast_tx, _server_handle, cancel_token) = rt.block_on(async {
            let worlds = repeat_with(|| World::new(config.clone()))
                            .take(config.worlds_parellised as usize)
                            .collect();

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

            PyResult::Ok((worlds, bcast_tx, server_handle, cancel_token))
        })?;

        Ok(Simulation {
            worlds,
            headless: is_headless,
            bcast_tx,
            _server_handle,
            cancel_token,
            rt,
        })
    }

    pub fn step(&mut self, world_id: i32, actions_i32: Vec<i32>) -> PyResult<(Vec<Vec<f32>>, Vec<f32>)> {

        let world = &mut self.worlds[world_id as usize];
        let actions: Vec<Action> = ToActions::to_actions(actions_i32);

        let rewards = world.apply_actions(actions);
        let obs = world.get_agents_observations();

        let world_state_with_id = serde_json::json!({ "world_id": world_id, "world_state": world.get_state() });
        let json_state = serde_json::to_string(&world_state_with_id)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("failed to serialize world state: {}", e)))?;

        if !self.headless {
            if let Err(_) = self.bcast_tx.send(json_state) {
                // println!("[DEBUG] Broadcast error: {}", e);
            }
        }
        Ok((obs, rewards))
    }

    pub fn shutdown(&self) -> PyResult<()> {
        println!("Shutting down simulation server...");
        self.cancel_token.cancel();
        Ok(())
    }

    pub fn reset(&mut self, world_id: i32) -> PyResult<Vec<Vec<f32>>> {
        
        let world = &mut self.worlds[world_id as usize];

        world.reset_agents();
        world.reset_stations();

        let obs = world.get_agents_observations();

        Ok(obs)
    }
}