use std::vec::Vec;

use serde::{Serialize};

use crate::agent::{ACTION_COUNT, Action, Agent, AgentState};
use crate::station::{Station, StationState};
use crate::config::{AgentConfig, Config, StationConfig};
use crate::location::{Layout, Location, get_location};
use crate::resource::{RESOURCE_COUNT, one_hot_vector_from_resource};
use crate::grid_map::{Entity, GridMap};


#[derive(Serialize)]
pub struct WorldState {
    agents: Vec<AgentState>,
    stations: Vec<StationState>
}

struct WorldContext {
    width: u32,
    height: u32,
    agent_layout: Layout,
    station_layout: Layout,
    agent_configs: Vec<AgentConfig>,
    station_configs: Vec<StationConfig>,
    agent_station_visibility: f32,
    agent_agent_visibility: f32,
    max_inputs: u32,
    max_outputs: u32,
}

pub struct World {
    agents: Vec<Agent>,
    stations: Vec<Station>,
    map: GridMap,
    context: WorldContext
}

impl World {
    pub fn new(config: Config) -> Self {

        let width = config.arena_width;
        let height = config.arena_height;

        let max_inputs = config.agents.iter().map(|agent| agent.inputs.len()).max().unwrap_or(0);

        let mut world = World {
            agents: Vec::new(),
            stations: Vec::new(),
            map: GridMap::new(width, height),
            context: WorldContext {
                width,
                height,
                agent_station_visibility: config.agent_station_visibility,
                agent_agent_visibility: config.agent_agent_visibility,
                agent_layout: config.initial_agent_layout,
                station_layout: config.station_layout,
                agent_configs: config.agents,
                station_configs: config.stations,
                max_inputs: max_inputs as u32,
                max_outputs: 1
            }
        };

        world.populate_map();

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
             0
        ).into_iter();

        self.agents = self.context.agent_configs.iter().map(|agent: &AgentConfig| Agent::new(
            agent.id as i32,
            initial_agent_locations.next().unwrap(),
            agent.inputs.clone(),
            agent.output,
        )).collect();

    }

    pub fn get_number_of_agents(&self) -> usize {
        self.agents.len()
    }

    pub fn reset_stations(&mut self) {

        let mut station_locations = get_location(
            self.context.station_layout,
            self.context.width,
            self.context.height,
            self.context.station_configs.len() as u32,
            1
        ).into_iter();

        self.stations = self.context.station_configs.iter().map(|station: &StationConfig| Station::new(
            station.id as i32,
            station_locations.next().unwrap(),
            station.station_type.clone(),
            station.resource,
        )).collect();
    }

    
    fn populate_map(&mut self) {

        self.reset_agents();
        self.reset_stations();
        self.map.reset();

        for agent in &self.agents {
            self.map.add_agent(agent.id, agent.location);
        }
        for station in &self.stations {
            self.map.add_station(station.get_id(), station.get_location().clone());
        }
    }

    pub fn apply_actions(&mut self, actions: Vec<Action>) {

        const INVALID_MOVE_REWARD: f32 = -0.02;

        for agent in &mut self.agents {
            agent.set_curr_reward(0.0);
        }


        for (index, action) in actions.into_iter().enumerate() {
            let original_location = self.agents[index].location.clone();
            let mut reward = -0.005; // Punishment for existing
            reward += match action {
                Action::MoveUp => {
                    self.agents[index].move_north();
                    self.map.move_agent(self.agents[index].id, original_location, self.agents[index].location);
                    if original_location == self.agents[index].location {
                        INVALID_MOVE_REWARD
                    } else {
                        0.0
                    }
                }
                Action::MoveDown => {
                    self.agents[index].move_south(self.context.height);
                    self.map.move_agent(self.agents[index].id, original_location, self.agents[index].location);
                    if original_location == self.agents[index].location {
                        INVALID_MOVE_REWARD
                    } else {
                        0.0
                    }
                }
                Action::MoveLeft => {
                    self.agents[index].move_west();
                    self.map.move_agent(self.agents[index].id, original_location, self.agents[index].location);
                    if original_location == self.agents[index].location {
                        INVALID_MOVE_REWARD
                    } else {
                        0.0
                    }
                }
                Action::MoveRight => {
                    self.agents[index].move_east(self.context.width);
                    self.map.move_agent(self.agents[index].id, original_location, self.agents[index].location);
                    if original_location == self.agents[index].location {
                        INVALID_MOVE_REWARD
                    } else {
                        0.0
                    }
                }
                Action::Interact => self.interact(index),
            };
            if let Some(desired_target) = self.get_known_desired_target(index) {
                reward += self.direction_reward(original_location, self.agents[index].location, desired_target);
            }
            if !self.map.has_location_been_visited(self.agents[index].location) {
                reward += 0.002;
                self.map.set_location_visited(self.agents[index].location);
            }

            // reward += self.agents[index].get_inventory_reward();

            self.agents[index].add_curr_reward(reward);
        }

    }

    pub fn spread_rewards(&mut self, alpha: f32) {
        let total_reward = self.agents.iter().map(|agent| agent.get_curr_reward()).sum::<f32>() / self.agents.len() as f32;
        for agent in &mut self.agents {
            let curr_reward = agent.get_curr_reward() * (1.0 - alpha) + total_reward * alpha;
            agent.set_curr_reward(curr_reward);
        }
    }


    fn interact(&mut self, agent_index: usize) -> f32 { // Returns reward

        let nearest_entity = self.get_nearest_visible_entity(&self.agents[agent_index]);
        // let nearest_entity = self.map.get_nearest_visible_entity(
        //     self.agents[agent_index].id,
        //     self.agents[agent_index].location,
        //     self.context.agent_visibility
        // );
        match nearest_entity {
            Some(Entity::Station(station_index)) => {
                let agent = &mut self.agents[agent_index];
                let station = &self.stations[station_index];
                agent.interact_with_station(station)
            }
            Some(Entity::Agent(other_agent_index)) => {
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

    pub fn get_agents_targets(&self) -> Vec<Vec<f32>> {
        // Target location and flag if it exists
        (0..self.agents.len()).map(|agent| {
            // if let Some(target) = self.get_possible_known_desired_target(agent) {
            if let Some(target) = self.get_known_desired_target(agent) {
                vec![
                    // (target.x - self.agents[agent].location.x) as f32 / self.context.width as f32,
                    // (target.y - self.agents[agent].location.y) as f32 / self.context.height as f32,
                    (target.x) as f32 / self.context.width as f32,
                    (target.y) as f32 / self.context.height as f32,
                    1.0
                ]
            } else {
                vec![0.0, 0.0, 0.0]
            }
        }).collect()
    }

    fn get_possible_known_desired_target(&self, agent_index: usize) -> Option<Location> {
        let agent = &self.agents[agent_index];

        // Get location closest to agent.location
        // agent.get_current_targets().into_iter().min_by_key(|target| {
        //     agent.location.distance_squared(target.location)
        // })

        let station_needed: Option<Location> = self.stations.iter().filter(|station| {
            agent.is_needed(Some(*station.get_resource())) && station.get_is_found()
        }).min_by_key(|station| {
            agent.location.distance_squared(*station.get_location())
        }).map(|station| *station.get_location());

        let agent_needed: Option<Location> = self.agents.iter().filter(|other_agent| {
            agent_index != other_agent.id as usize &&
            (
                agent.is_needed(other_agent.get_curr_output()) ||
                other_agent.is_needed(agent.get_curr_output())
            )
        }).min_by_key(|other_agent| {
            agent.location.distance_squared(other_agent.location)
        }).map(|other_agent| other_agent.location);

        // Return closest target coords
        match (station_needed, agent_needed) {
            (Some(station_needed), Some(agent_needed)) => {
                if agent.location.distance_squared(station_needed) < agent.location.distance_squared(agent_needed) {
                    Some(station_needed)
                } else {
                    Some(agent_needed)
                }
            }
            (Some(station_needed), None) => {
                Some(station_needed)
            }
            (None, Some(agent_needed)) => {
                Some(agent_needed)
            }
            (None, None) => {
                None
            }
        }

    }

    fn get_known_desired_target(&self, agent_index: usize) -> Option<Location> {
        let agent = &self.agents[agent_index];

        // Get location closest to agent.location
        agent.get_current_targets().into_iter().min_by_key(|target| {
            agent.location.distance_squared(target.location)
        }).map(|target| target.location)

    }

    fn direction_reward(&self, original_location: Location, new_location: Location, desired_target: Location) -> f32 {

        //! TODO: Is this wanted
        // if (desired_target - new_location).magnitude() <= self.context.agent_visibility as f32 {
        //     return 0.0;
        // }

        // Use cosine similarity - [-1, 1]
        let cosine_similarity = Location::cosine_similarity(original_location, new_location, desired_target);

        // Near 1 is the right direction
        cosine_similarity * 0.005
 
    }

    fn get_nearest_visible_entity(&self, agent: &Agent) -> Option<Entity> {

        let nearest_station = self.stations.iter().enumerate()
            .filter_map(|(index, t_station)| {
                let dist_sq = agent.location.distance_squared(*t_station.get_location());
                if dist_sq <= (self.context.agent_station_visibility as u64).pow(2) {
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
                if dist_sq <= (self.context.agent_agent_visibility as u64).pow(2) {
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

    fn get_agent_observation(&self, agent_index: usize) -> Vec<f32> {

        const BASIC_INPUT_TENSOR_SIZE: u32 = 3;
        const OBS_PER_INVENTORY_PIECE: u32 = 2 + RESOURCE_COUNT as u32;
        let max_targets = self.context.max_inputs + self.context.max_outputs;
        let other_obs_size = BASIC_INPUT_TENSOR_SIZE + OBS_PER_INVENTORY_PIECE * max_targets;

        const SELF_AGENT_OBS_SIZE: u32 = 2;
        const TARGET_OBS_SIZE: u32 = 6 + RESOURCE_COUNT as u32;
        let total_obs_size: u32 = SELF_AGENT_OBS_SIZE + TARGET_OBS_SIZE * max_targets + other_obs_size;

        let mut obs = Vec::with_capacity(total_obs_size as usize);

        let agent = &self.agents[agent_index];
        
        // Agent Location (0-1)
        obs.push(agent.location.x as f32 / self.context.width as f32);
        obs.push(agent.location.y as f32 / self.context.height as f32);

        // Agent's own hardcoded memory of the world
        for i in 0..max_targets as usize {
            if i < agent.agent_targets.len() {
                // Target's resource
                obs.extend(one_hot_vector_from_resource(agent.agent_targets[i].resource));

                // Has the agent come across this station/agent
                obs.push(agent.agent_targets[i].is_found as i32 as f32);

                // station/agent last known location - relative
                obs.push(((agent.agent_targets[i].location.x - agent.location.x) as f32 + self.context.width as f32) / (2.0 * self.context.width as f32));
                obs.push(((agent.agent_targets[i].location.y - agent.location.y) as f32 + self.context.height as f32) / (2.0 * self.context.height as f32));
                // If we have it, was it a static station or a moving agent
                obs.push(f32::from(&agent.agent_targets[i].station_type));
                
                // Do we currently need the resource
                obs.push(agent.agent_targets[i].is_current_target as i32 as f32);

                // Does agent have the resource
                obs.push(agent.agent_targets[i].is_collected as i32 as f32);
            } else {
                obs.extend(vec![0.0; TARGET_OBS_SIZE as usize]);
            }
        }

        // Next to which station/agent if any
        let nearest_entity = self.get_nearest_visible_entity(agent);
        // let nearest_entity = self.map.get_nearest_visible_entity(
        //     agent.id,
        //     agent.location,
        //     self.context.agent_visibility
        // );

        let entity_obs = match nearest_entity {
            Some(Entity::Station(station_index)) => {
                let t_station = &self.stations[station_index];
                let mut station_obs = t_station.get_self_observations(self.context.width, self.context.height);
                // Add padding
                station_obs.resize(other_obs_size as usize, 0.0);

                // Set is found
                t_station.set_found();

                station_obs
            }
            Some(Entity::Agent(agent_index)) => {
                let t_agent = &self.agents[agent_index];
                t_agent.get_self_observations(self.context.width, self.context.height, max_targets)
            }
            None => vec![0.0; other_obs_size as usize]
        };

        assert!(entity_obs.len() as u32 == other_obs_size);

        obs.extend(entity_obs);
        
        assert!(obs.len() as u32 == total_obs_size);

        obs
    }

    pub fn get_agent_action_count(&self) -> u32 {
        ACTION_COUNT
    }

    pub fn get_agent_obs_size(&self) -> u32 {
        
        const BASIC_INPUT_TENSOR_SIZE: u32 = 3;
        const OBS_PER_INVENTORY_PIECE: u32 = 2 + RESOURCE_COUNT as u32;
        let target_size = self.context.max_inputs + self.context.max_outputs;
        let obs_size = BASIC_INPUT_TENSOR_SIZE + OBS_PER_INVENTORY_PIECE * target_size;

        const SELF_AGENT_OBS_SIZE: u32 = 2;
        const TARGET_OBS_SIZE: u32 = 6 + RESOURCE_COUNT as u32;
        
        SELF_AGENT_OBS_SIZE + TARGET_OBS_SIZE * target_size + obs_size

    }

    pub fn get_agent_obs_mask(&self) -> Vec<bool> {
        // True is for discrete obs
        // False is for continuous obs
        let mut obs_mask = vec![true; self.get_agent_obs_size() as usize];

        const SELF_OBS: usize = 2;

        // Self Coords
        obs_mask[0] = false;
        obs_mask[1] = false;

        let max_targets = self.context.max_inputs + self.context.max_outputs;
        const TARGET_OBS_SIZE: usize = 6 + RESOURCE_COUNT;
        // self targets coords
        for i in 0..max_targets {
            obs_mask[SELF_OBS + RESOURCE_COUNT as usize + 1 + i as usize * TARGET_OBS_SIZE] = false;
            obs_mask[SELF_OBS + RESOURCE_COUNT as usize + 2 + i as usize * TARGET_OBS_SIZE] = false;
        }

        let previous_obs_size = SELF_OBS + (max_targets as usize * TARGET_OBS_SIZE);

        obs_mask[previous_obs_size + 1] = false;
        obs_mask[previous_obs_size + 2] = false;

        obs_mask

    }

    pub fn get_agent_obs(&self, agent_index: usize) -> Vec<f32> {
        self.get_agent_observation(agent_index)
    }

    pub fn get_agents_obs(&self) -> Vec<Vec<f32>> {
        (0..self.agents.len()).map(|agent| self.get_agent_observation(agent)).collect()

    }

    pub fn get_agent_reward(&self, agent_index: usize) -> f32 {
        self.agents[agent_index].get_curr_reward()
    }

    pub fn get_agents_reward(&self) -> Vec<f32> {
        self.agents.iter().map(|agent| agent.get_curr_reward()).collect()
    }

    pub fn get_global_obs_size(&self) -> u32 {
        let max_targets = self.context.max_inputs + self.context.max_outputs;
        let target_obs_size = RESOURCE_COUNT as u32 + 2;

        self.get_number_of_agents() as u32 * (
            2 // location
            + target_obs_size * max_targets // targets
            + self.get_number_of_agents() as u32 * ( // Other Agents
                2 // location
                + target_obs_size * max_targets // targets
            )
            + self.context.station_configs.len() as u32 * (
                2 // location
                + RESOURCE_COUNT as u32 // resource
                + 1 // station type
            )
        )
        
    }

    pub fn get_global_obs(&self) -> Vec<f32> {
        let mut obs = Vec::new();

        let max_targets = self.context.max_inputs + self.context.max_outputs;
        let target_obs_size = RESOURCE_COUNT + 2;

        // for every agent, give all position, and all relative positions to every agent and station.
        for agent in &self.agents {
            let loc = agent.get_location();
            obs.push(loc.x as f32 / self.context.width as f32);
            obs.push(loc.y as f32 / self.context.height as f32);

            for target in 0..max_targets {
                if target < agent.agent_targets.len() as u32 {
                    // What is in target
                    obs.extend(one_hot_vector_from_resource(agent.agent_targets[target as usize].resource));

                    // Do we have it
                    obs.push(agent.agent_targets[target as usize].is_collected as i32 as f32);

                    // Do we want it, or do we want to drop it off
                    obs.push(f32::from(&agent.agent_targets[target as usize].station_type));
                } else {
                    obs.extend(vec![0.0; target_obs_size as usize]);
                }

            }

            for other_agent in &self.agents {
                let other_loc = other_agent.get_location();
                let vector = other_loc - loc;
                obs.push(vector.x as f32 / self.context.width as f32);
                obs.push(vector.y as f32 / self.context.height as f32);

                for target in 0..max_targets {
                    if target < other_agent.agent_targets.len() as u32 {
                        // What is in target
                        obs.extend(one_hot_vector_from_resource(other_agent.agent_targets[target as usize].resource));

                        // Does the agent have it
                        obs.push(other_agent.agent_targets[target as usize].is_collected as i32 as f32);

                        // Do we want it, or do we want to drop it off
                        obs.push(f32::from(&other_agent.agent_targets[target as usize].station_type));
                    } else {
                        obs.extend(vec![0.0; target_obs_size as usize]);
                    }

                }
            }

            for station in &self.stations {
                let station_loc = station.get_location();
                let vector = station_loc - loc;
                obs.push(vector.x as f32 / self.context.width as f32);
                obs.push(vector.y as f32 / self.context.height as f32);

                obs.extend(one_hot_vector_from_resource(*station.get_resource()));
                obs.push(*station.get_station_type() as i32 as f32);

            }
        }

        assert_eq!(obs.len() as u32, self.get_global_obs_size());

        obs
    }

}