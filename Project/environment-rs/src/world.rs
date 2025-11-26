use std::vec::Vec;

use serde::{Serialize};

use crate::agent::{Action, Agent, AgentState};
use crate::station::{Station, StationState};
use crate::config::{AgentConfig, Config, StationConfig};
use crate::location::{Layout, Location, get_location};


#[derive(Serialize)]
pub struct WorldState {
    agents: Vec<AgentState>,
    stations: Vec<StationState>
}

struct WorldContext {
    width: u32,
    height: u32,
    agent_size: u32,
    station_size: u32,
    agent_layout: Layout,
    station_layout: Layout,
    agent_configs: Vec<AgentConfig>,
    station_configs: Vec<StationConfig>,
    agent_visibility: u32,
    max_inputs: u32,
    max_outputs: u32,
}

pub struct World {
    agents: Vec<Agent>,
    stations: Vec<Station>,
    context: WorldContext
}

pub enum Entity {
    Station(usize),
    Agent(usize),
}

impl World {
    pub fn new(config: Config) -> Self {

        let width = config.arena_width;
        let height = config.arena_height;

        let max_inputs = config.agents.iter().map(|agent| agent.inputs.len()).max().unwrap_or(0);

        let mut world = World {
            agents: Vec::new(),
            stations: Vec::new(),
            context: WorldContext {
                width,
                height,
                agent_size: config.agent_size,
                station_size: config.station_size,
                agent_layout: config.initial_agent_layout,
                station_layout: config.station_layout,
                agent_configs: config.agents,
                station_configs: config.stations,
                agent_visibility: config.agent_size * 6,
                max_inputs: max_inputs as u32,
                max_outputs: 1
            }
        };

        world.reset_agents();
        world.reset_stations();

        world
    }

    pub fn get_state(&self) -> WorldState {
        WorldState {
            agents: self.agents.iter().map(|agent| agent.into()).collect(),
            stations: self.stations.iter().map(|station| station.into()).collect(),
        }
    }

    pub fn reset_agents(&mut self) {
        
        let mut initial_agent_locations = get_location(
            self.context.agent_layout,
            self.context.width, self.context.height,
            self.context.agent_configs.len() as u32,
            self.context.agent_size * 3
        ).into_iter();

        self.agents = self.context.agent_configs.iter().map(|agent: &AgentConfig| Agent::new(
            agent.id as i32,
            initial_agent_locations.next().unwrap(),
            agent.inputs.clone(),
            agent.output,
        )).collect();

    }

    pub fn reset_stations(&mut self) {

        let mut station_locations = get_location(
            self.context.station_layout,
            self.context.width,
            self.context.height,
            self.context.station_configs.len() as u32,
            self.context.station_size * 3
        ).into_iter();

        self.stations = self.context.station_configs.iter().map(|station: &StationConfig| Station::new(
            station.id as i32,
            station_locations.next().unwrap(),
            station.station_type.clone(),
            station.resource,
        )).collect();
    }

    pub fn apply_actions(&mut self, actions: Vec<Action>) -> Vec<f32> {

        let mut rewards = Vec::new();
        for (index, action) in actions.into_iter().enumerate() {
            let original_location = self.agents[index].location.clone();
            let mut reward = -0.01; // Punishment for existing
            reward += match action {
                Action::MoveUp => {
                    self.agents[index].move_north();
                    if original_location == self.agents[index].location {
                        -5.0
                    } else {
                        0.0
                    }
                }
                Action::MoveDown => {
                    self.agents[index].move_south(self.context.height);
                    if original_location == self.agents[index].location {
                        -5.0
                    } else {
                        0.0
                    }
                }
                Action::MoveLeft => {
                    self.agents[index].move_west();
                    if original_location == self.agents[index].location {
                        -5.0
                    } else {
                        0.0
                    }
                }
                Action::MoveRight => {
                    self.agents[index].move_east(self.context.width);
                    if original_location == self.agents[index].location {
                        -5.0
                    } else {
                        0.0
                    }
                }
                Action::Interact => self.interact(index),
            };
            if let Some(desired_location) = self.get_desired_target(index) {
                reward += self.direction_reward(original_location, self.agents[index].location, desired_location);
            }

            rewards.push(reward);
        }

        rewards
    }

    fn interact(&mut self, agent_index: usize) -> f32 { // Returns reward

        let nearest_entity = self.get_nearest_visible_entity(&self.agents[agent_index]);
        match nearest_entity {
            Some(Entity::Station(station_index)) => {
                let agent = &mut self.agents[agent_index];
                let station = &self.stations[station_index];
                agent.interact_with_station(station)
            }
            Some(Entity::Agent(other_agent_index)) => {
                assert_ne!(agent_index, other_agent_index); // Should never be returned by nearest_entity

                // Splitting the agent array to borrow both required agents mutably
                let (agent, other_agent) = if agent_index < other_agent_index {
                    let (left, right) = self.agents.split_at_mut(other_agent_index);
                    (&mut left[agent_index], &mut right[0])
                } else {
                    let (left, right) = self.agents.split_at_mut(agent_index);
                    (&mut right[0], &mut left[other_agent_index])
                };

                agent.interact_with_agent(other_agent)
            }
            None => 0.0
        }
    }

    fn get_desired_target(&self, agent_index: usize) -> Option<Location> {
        let agent = &self.agents[agent_index];

        // Get location closest to agent.location
        agent.get_current_target_locations().into_iter().min_by_key(|location| {
            agent.location.distance_squared(*location)
        })

    }

    // fn position_reward(&self, agent_index: usize) -> f32 {
    //     let agent = &self.agents[agent_index];
    //     let x = agent.location.x;
    //     let y = agent.location.y;
    //     if x >= 0 && x <= self.context.width as i32 && y > 0 && y < self.context.height as i32 {
    //         0.0
    //     } else {
    //         -5000.0
    //     }
    // }

    fn direction_reward(&self, original_location: Location, new_location: Location, desired_target: Location) -> f32 {

        if (desired_target - new_location).magnitude() <= self.context.agent_visibility as f32 {
            return 0.0;
        }

        // Use cosine similarity - [-1, 1]
        let cosine_similarity = Location::cosine_similarity(original_location, new_location, desired_target);

        // Near 1 is the right direction
        cosine_similarity
 
    }

    fn get_nearest_visible_entity(&self, agent: &Agent) -> Option<Entity> {

        let nearest_station = self.stations.iter().enumerate()
            .filter_map(|(index, t_station)| {
                let dist_sq = agent.location.distance_squared(t_station.location);
                if dist_sq <= (self.context.agent_visibility as u64).pow(2) {
                    Some((index, dist_sq))
                } else {
                    None
                }
            })
            .min_by_key(|(_, dist_sq)| *dist_sq);
            
        let nearest_agent = self.agents.iter().enumerate()
            .filter(|(_, t_agent)| t_agent.id != agent.id) // Filter out self
            .filter_map(|(index, t_agent)| {
                let dist_sq = agent.location.distance_squared(t_agent.location);
                if dist_sq <= self.context.agent_visibility as u64 {
                    Some((index, dist_sq))
                } else {
                    None
                }
            })
            .min_by_key(|(_, dist_sq)| *dist_sq);

        match (nearest_station, nearest_agent) {
            // Both exist, we must compare them
            (Some((station_index, station_dist)), Some((agent_index, agent_dist))) => {
                if station_dist < agent_dist {
                    // Station is closest
                    Some(Entity::Station(station_index))
                } else {
                    // Agent is closest (or equal)
                    Some(Entity::Agent(agent_index))
                }
            },
            // Only station exists
            (Some((station_index, _)), None) => {
                // Station is closest
                Some(Entity::Station(station_index))
            },
            // Only agent exists
            (None, Some((agent_index, _))) => {
                // Agent is closest
                Some(Entity::Agent(agent_index))
            },
            // Neither exists
            (None, None) => {
                // No entity visible
                None
            }
        }
    }

    fn get_agent_observation(&self, agent: &Agent) -> Vec<f32> {

        const BASIC_INPUT_TENSOR_SIZE: u32 = 3;
        const OBS_PER_INVENTORY_PIECE: u32 = 3;
        let target_size = self.context.max_inputs + self.context.max_outputs;
        let obs_size = BASIC_INPUT_TENSOR_SIZE + OBS_PER_INVENTORY_PIECE * target_size;

        let mut obs = Vec::new();
        
        // Agent Location (0-1)
        obs.push(agent.location.x as f32 / self.context.width as f32);
        obs.push(agent.location.y as f32 / self.context.height as f32);

        // Agent's own hardcoded memory of the world
        for i in 0..target_size as usize {
            if i < agent.agent_targets.len() {
                // Target's resource
                obs.push(f32::from(&agent.agent_targets[i].resource));

                // Has the agent come across this station/agent
                obs.push(agent.agent_targets[i].is_found as i32 as f32);

                // station/agent last known location - relative
                obs.push((agent.agent_targets[i].location.x - agent.location.x) as f32 / self.context.width as f32);
                obs.push((agent.agent_targets[i].location.y - agent.location.y) as f32 / self.context.height as f32);
                // If we have it, was it a static station or a moving agent
                obs.push(f32::from(&agent.agent_targets[i].station_type));
                
                // Do we currently need the resource
                obs.push(agent.agent_targets[i].is_current_target as i32 as f32);

                // Does agent have the resource
                obs.push(agent.agent_targets[i].is_collected as i32 as f32);
            } else {
                obs.push(0.0);
                obs.push(0.0);
                obs.push(0.0);
                obs.push(0.0);
                obs.push(0.0);
                obs.push(0.0);
                obs.push(0.0);
            }
        }

        // Next to which station/agent if any
        let nearest_entity = self.get_nearest_visible_entity(agent);
        let entity_obs = match nearest_entity {
            Some(Entity::Station(station_index)) => {
                let t_station = &self.stations[station_index];
                let mut station_obs = t_station.get_self_observations(self.context.width, self.context.height);
                // Add padding
                station_obs.resize(obs_size as usize, 0.0);
                station_obs
            }
            Some(Entity::Agent(agent_index)) => {
                let t_agent = &self.agents[agent_index];
                t_agent.get_self_observations(self.context.width, self.context.height, target_size)
            }
            None => vec![0.0; obs_size as usize]
        };

        assert!(entity_obs.len() as u32 == obs_size);

        obs.extend(entity_obs);

        obs
    }

    pub fn get_agents_observations(&self) -> Vec<Vec<f32>> {
        self.agents.iter().map(|agent| self.get_agent_observation(agent)).collect()
    }

}