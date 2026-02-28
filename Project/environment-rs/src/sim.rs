use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray1};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use tokio::runtime::Runtime;
use serde_json::{self, Value};
use tokio::signal;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use crate::agent::{ToActions};
use crate::config::{load_config};
use crate::world::World;
use crate::websocket::websocket;
use tokio::task::JoinHandle;
use std::iter::repeat_with;

#[pyclass(skip_from_py_object)]
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
            _server_handle,
            cancel_token,
            rt,
            config: SimConfig {
                headless: is_headless,
                n_agents: config.agents.len() as u32,
            }
        })
    }

    pub fn parallel_step<'py>(
        &mut self,
        py: Python<'py>,
        flat_actions: Vec<i32>,
        flat_world_comms: Vec<f32>
    ) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
            
        let n_agents = self.worlds[0].get_number_of_agents();
        let n_worlds = self.worlds.len();
        let world_comm_length = flat_world_comms.len() / n_worlds;
        
        if flat_actions.len() != n_worlds * n_agents {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>("Invalid number of actions"));
        }

        let obs_size = self.worlds[0].get_agent_obs_size() as usize + world_comm_length;
        let total_obs_len = n_worlds * n_agents * obs_size;
        let total_global_obs_len = n_worlds * self.worlds[0].get_global_obs_size() as usize;
        let total_rew_len = n_worlds * n_agents;

        let results: Vec<(Vec<f32>, Vec<f32>, Vec<f32>, Option<Value>)> = self.worlds.par_iter_mut()
            .enumerate()
            .zip(flat_actions.par_chunks(n_agents))
            .zip(flat_world_comms.par_chunks(world_comm_length))
            .map(|(((world_id, world), world_actions), world_comms)| {

                let agent_actions = ToActions::to_actions(world_actions.to_vec());

                world.apply_actions(agent_actions);
                world.spread_rewards(0.4);
                
                let obs = world.get_agents_obs();
                let mut global_obs = world.get_global_obs();
                let rewards = world.get_agents_reward();

                let mut flat_world_obs = Vec::with_capacity(obs.len() * (obs[0].len() + world_comms.len()));

                for mut agent_obs in obs {
                    agent_obs.extend_from_slice(world_comms);
                    flat_world_obs.extend_from_slice(&agent_obs);
                }

                global_obs.extend_from_slice(world_comms);

                let json_state = if !self.config.headless { 
                    Some(serde_json::json!({ "world_id": world_id as f32, "world_state": world.get_state() }))
                } else { 
                    None
                };

                (flat_world_obs, global_obs, rewards, json_state)
            })
            .collect();
            
        let mut all_obs = Vec::with_capacity(total_obs_len);
        let mut all_global_obs = Vec::with_capacity(total_global_obs_len);
        let mut all_rewards = Vec::with_capacity(total_rew_len);

        for (w_obs, w_global_obs, w_rew, json_state) in results {
            all_obs.extend(w_obs);
            all_global_obs.extend(w_global_obs);
            all_rewards.extend(w_rew);

            if let Some(state) = json_state {
                let state_string = serde_json::to_string(&state)
                        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("failed to serialize world state: {}", e)))?;
                let _ = self.bcast_tx.send(state_string);
            }
        }

        let obs_numpy: Bound<'py, PyArray1<f32>> = all_obs.into_pyarray(py);
        let global_obs_numpy: Bound<'py, PyArray1<f32>> = all_global_obs.into_pyarray(py);
        let rewards_numpy: Bound<'py, PyArray1<f32>> = all_rewards.into_pyarray(py);

        Ok((obs_numpy, global_obs_numpy, rewards_numpy))

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

    pub fn get_all_global_obs(&self, py: Python<'_>,  flat_world_comms: Vec<f32>) -> PyResult<Vec<f32>> {
        let world_comm_length = flat_world_comms.len() / self.worlds.len();

        Python::detach(py, || {
            let results: Vec<f32> = self.worlds.par_iter()
                .zip(flat_world_comms.par_chunks(world_comm_length))
                .flat_map(|(world, world_comms)| {
                    let mut obs = world.get_global_obs();
                    
                    obs.extend_from_slice(world_comms);
                    
                    obs
                })
                .collect();
            
            Ok::<Vec<f32>, PyErr>(results)
        })

    }

    pub fn get_global_obs_size(&self, world_id: i32) -> PyResult<u32> {
        let world = &self.worlds[world_id as usize];
        let obs_size = world.get_global_obs_size();
        Ok(obs_size)
    }


}