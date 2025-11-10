use std::vec::Vec;

use serde::{Serialize};

use crate::agent::{Action, Agent, AgentState};
use crate::station::{Station, StationState};
use crate::config::{AgentConfig, Config, StationConfig};
use crate::location::{Layout, get_location};


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
                agent_visibility: config.agent_size * 5,
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
            let mut reward = match action {
                Action::MoveUp => {
                    self.agents[index].move_north();
                    0.0
                }
                Action::MoveDown => {
                    self.agents[index].move_south();
                    0.0
                }
                Action::MoveLeft => {
                    self.agents[index].move_west();
                    0.0
                }
                Action::MoveRight => {
                    self.agents[index].move_east();
                    0.0
                }
                Action::Interact => self.interact(index),
            };
            reward += self.position_reward(index);
            rewards.push(reward);
        }

        rewards
    }

    // add rewards on movement towards correct station/aghent

    fn interact(&mut self, agent_index: usize) -> f32 { // Returns reward

        let mut agent = self.agents[agent_index].clone();
        let nearest_entity = self.get_nearest_visible_entity(&agent);
        match nearest_entity {
            Some(Entity::Station(station_index)) => {
                let station = &self.stations[station_index];
                agent.interact_with_station(station)
            }
            Some(Entity::Agent(agent_index)) => {
                let t_agent = &mut self.agents[agent_index];
                agent.interact_with_agent(t_agent)
            }
            None => 0.0,
        }
    }

    fn position_reward(&self, agent_index: usize) -> f32 {
        let agent = &self.agents[agent_index];
        let x = agent.location.x;
        let y = agent.location.y;
        if x > 0 && x < self.context.width as i32 && y > 0 && y < self.context.height as i32 {
            0.0
        } else {
            -50.0
        }
    }


    fn get_nearest_visible_entity(&self, agent: &Agent) -> Option<Entity> {

        let nearest_station = self.stations.iter().enumerate()
            .filter_map(|(index, t_station)| {
                let dist_sq = agent.location.distance_squared(t_station.location);
                if dist_sq <= self.context.agent_visibility as u64 {
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

        let mut obs = Vec::new();
        
        // Agent Location (0-1)
        obs.push(agent.location.x as f32 / self.context.width as f32);
        obs.push(agent.location.y as f32 / self.context.height as f32);


        // Next to which station/agent if any
        let nearest_entity = self.get_nearest_visible_entity(agent);
        let entity_obs = match nearest_entity {
            Some(Entity::Station(station_index)) => {
                let t_station = &self.stations[station_index];
                t_station.get_self_observations(self.context.width, self.context.height)
            }
            Some(Entity::Agent(agent_index)) => {
                let t_agent = &self.agents[agent_index];
                t_agent.get_self_observations(self.context.width, self.context.height)
            }
            None => vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };

        assert!(entity_obs.len() == 7);

        obs.extend(entity_obs);

        obs
    }

    pub fn get_agents_observations(&self) -> Vec<Vec<f32>> {
        self.agents.iter().map(|agent| self.get_agent_observation(agent)).collect()
    }

}