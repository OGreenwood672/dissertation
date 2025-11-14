use std::collections::HashSet;
use std::vec::Vec;

use serde::{Serialize};

use crate::agent::{Action, Agent, AgentState};
use crate::station::{Station, StationState, StationType};
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
                agent_visibility: config.agent_size * 25,
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
            if let Some(desired_location) = self.get_desired_target(index) {
                reward += self.direction_reward(original_location, self.agents[index].location, desired_location);
            }
            reward += self.position_reward(index);
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

        // Dropoff an Item
        if let Some(agent_inventory) = agent.inventory {
            if agent_inventory == agent.output {
                for station in &self.stations {
                    if station.station_type == StationType::DropOff && station.resource == agent.output {
                        return Some(station.location);
                    }
                }
                for other_agent in &self.agents {
                    if other_agent.id != agent.id {
                        if let Some(other_inventory) = other_agent.inventory {   
                            if other_inventory == agent.output {
                                return Some(other_agent.location);
                            }
                        }
                    }
                }
            }
        } else {
            let inventory_set = match agent.inventory {
                Some(inventory) => HashSet::from([inventory]),
                None => HashSet::new(),
            };
            let inputs_required: HashSet<_> = agent.input
                .difference(&inventory_set)
                .copied() 
                .collect();
            for station in &self.stations {
                if station.station_type == StationType::PickUp && inputs_required.contains(&station.resource) {
                    return Some(station.location);
                }
            }
            for other_agent in &self.agents {
                if other_agent.id != agent.id {
                    if let Some(other_inventory) = other_agent.inventory {
                        if inputs_required.contains(&other_inventory) {
                            return Some(other_agent.location);
                        }
                    }
                }
            }
        
        }

        None

    }

    fn position_reward(&self, agent_index: usize) -> f32 {
        let agent = &self.agents[agent_index];
        let x = agent.location.x;
        let y = agent.location.y;
        if x > 0 && x < self.context.width as i32 && y > 0 && y < self.context.height as i32 {
            0.0
        } else {
            -5000.0
        }
    }

    fn direction_reward(&self, original_location: Location, new_location: Location, desired_target: Location) -> f32 {

        if (desired_target - new_location).magnitude() <= self.context.agent_visibility as f32 {
            return 0.0;
        }

        // Use cosine similarity - [-1, 1]
        let cosine_similarity = Location::cosine_similarity(original_location, new_location, desired_target);

        // Near 1 is the right direction
        cosine_similarity * 10.0
 
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