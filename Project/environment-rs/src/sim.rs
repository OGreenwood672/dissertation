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
#[derive(Clone)]
pub struct SimConfig {
    headless: bool,

    #[pyo3(get)]
    n_agents: u32,
}


#[pyclass]
pub struct Simulation {
    worlds: Vec<World>,
    bcast_tx: broadcast::Sender<String>,
    bcast_is_erring: bool,
    _server_handle: JoinHandle<()>,
    cancel_token: CancellationToken,
    #[expect(unused)]
    rt: Runtime,

    #[pyo3(get)]
    config: SimConfig
}

#[pymethods]
impl Simulation {
    #[new]
    pub fn new(config_path: String, worlds_parallised: i32) -> PyResult<Self> {

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
                            .take(worlds_parallised as usize)
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
            bcast_tx,
            bcast_is_erring: false,
            _server_handle,
            cancel_token,
            rt,
            config: SimConfig {
                headless: is_headless,
                n_agents: config.agents.len() as u32,
            }
        })
    }

    pub fn step(&mut self, world_id: i32, actions_i32: Vec<i32>) -> PyResult<()> {

        let world = &mut self.worlds[world_id as usize];
        let actions: Vec<Action> = ToActions::to_actions(actions_i32);

        world.apply_actions(actions);

        if !self.config.headless {
            
            let world_state_with_id = serde_json::json!({ "world_id": world_id, "world_state": world.get_state() });
            let json_state = serde_json::to_string(&world_state_with_id)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("failed to serialize world state: {}", e)))?;

            if let Err(e) = self.bcast_tx.send(json_state) {
                if !self.bcast_is_erring {
                    println!("[DEBUG] Broadcast error: {}", e);
                    self.bcast_is_erring = true;
                }
            } else {
                if self.bcast_is_erring {
                    println!("[DEBUG] Broadcast recovered");
                    self.bcast_is_erring = false;
                }
            }
        }
        Ok(())
    }

    pub fn shutdown(&self) -> PyResult<()> {
        println!("Shutting down simulation server...");
        self.cancel_token.cancel();
        Ok(())
    }

    pub fn reset(&mut self, world_id: i32) -> PyResult<()> {
        
        let world = &mut self.worlds[world_id as usize];

        world.reset_agents();
        world.reset_stations();

        Ok(())
    }

    pub fn get_number_of_agents(&self, world_id: i32) -> PyResult<usize> {
        let world = &self.worlds[world_id as usize];
        Ok(world.get_number_of_agents())
    }

    pub fn get_agent_action_count(&self, world_id: i32) -> PyResult<u32> {
        let world = &self.worlds[world_id as usize];
        Ok(world.get_agent_action_count())
    }

    pub fn get_agent_obs(&self, world_id: i32, agent_id: i32) -> PyResult<Vec<f32>> {
        let world = &self.worlds[world_id as usize];
        let obs = world.get_agent_obs(agent_id as usize);
        Ok(obs)
    }

    pub fn get_agent_obs_size(&self, world_id: i32) -> PyResult<u32> {
        let world = &self.worlds[world_id as usize];
        let obs_size = world.get_agent_obs_size();
        Ok(obs_size)
    }

    pub fn get_agent_reward(&self, world_id: i32, agent_id: i32) -> PyResult<f32> {
        let world = &self.worlds[world_id as usize];
        let obs = world.get_agent_reward(agent_id as usize);
        Ok(obs)
    }

    pub fn get_global_obs(&self, world_id: i32) -> PyResult<Vec<f32>> {
        let world = &self.worlds[world_id as usize];
        let obs = world.get_global_obs();
        Ok(obs)
    }

    pub fn get_global_obs_size(&self, world_id: i32) -> PyResult<u32> {
        let world = &self.worlds[world_id as usize];
        let obs_size = world.get_global_obs_size();
        Ok(obs_size)
    }


}