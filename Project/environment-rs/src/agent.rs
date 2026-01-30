use std::collections::HashSet;
use std::convert::From;
use std::vec::Vec;
use serde::Serialize;

use crate::location::Location;
use crate::resource::ResourceType;
use crate::station::{Station, StationType};

#[derive(Clone, PartialEq)]
pub struct AgentTarget {
    pub is_station: bool,
    pub station_type: StationType,
    pub resource: ResourceType,
    pub location: Location,
    pub is_found: bool,
    pub is_current_target: bool,
    pub is_collected: bool,
}

impl AgentTarget {
    pub fn new(station_type: StationType, resource: ResourceType) -> Self {
        AgentTarget {
            is_station: false,
            station_type,
            resource,
            location: Location { x: -1, y: -1 },
            is_found: false,
            is_current_target: false,
            is_collected: false,
        }
    }

    pub fn set_found(&mut self, location: Location, is_station: bool) {
        self.location = location;
        self.is_found = true;
        self.is_station = is_station;
    }

    pub fn set_current_target(&mut self) {
        self.is_current_target = true;
    }

    pub fn collect(&mut self) {
        self.is_current_target = false;
        self.is_collected = true;
    }

    pub fn dropoff(&mut self) {
        self.is_collected = false;
        self.is_current_target = false;
    }

}

#[derive(Serialize)]
pub struct AgentState {
    pub id: i32,
    pub location: Location,
}

#[derive(Clone)]
pub struct Agent {
    pub id: i32,
    pub location: Location,
    pub curr_reward: f32,
    pub agent_targets: Vec<AgentTarget>,
}

// actions: move up, down, left, right, interact, 
pub const ACTION_COUNT: u32 = 5;
pub enum Action {
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    Interact
}

impl From<i32> for Action {
    fn from(item: i32) -> Self {
        match item {
            0 => Action::MoveUp,
            1 => Action::MoveDown,
            2 => Action::MoveLeft,
            3 => Action::MoveRight,
            4 => Action::Interact,
            _ => panic!("Invalid action value"),
        }
    }
}

pub trait ToActions {
    fn to_actions(self) -> Vec<Action>;
}

impl ToActions for Vec<i32> {
    fn to_actions(self) -> Vec<Action> {
        self.into_iter().map(Action::from).collect()
    }

}

impl Agent {
    /// Creates a new Agent instance.
    pub fn new(id: i32, location: Location, input: HashSet<ResourceType>, output: ResourceType) -> Self {

        let mut agent_targets: Vec<AgentTarget> = input.into_iter().map(|input| {
            AgentTarget::new(StationType::PickUp, input)
        }).collect();
        agent_targets.push(AgentTarget::new(StationType::DropOff, output));

        let mut agent = Agent {
            id,
            location,
            curr_reward: 0.0,
            agent_targets,
        };
        agent.reset_targets();

        agent
    }

    /// Moves the agent one unit north.
    pub fn move_north(&mut self) {
        if self.location.y > 0 as i32 {
            self.location.y = self.location.y - 1;
        }
    }

    /// Moves the agent one unit south.
    pub fn move_south(&mut self, height: u32) {
        if self.location.y < height as i32 - 1 {
            self.location.y = self.location.y + 1;
        }
    }

    /// Moves the agent one unit west.
    pub fn move_west(&mut self) {
        if self.location.x > 0 {
            self.location.x = self.location.x - 1;
        }
    }

    /// Moves the agent one unit east.
    pub fn move_east(&mut self, width: u32) {
        if self.location.x < width as i32 - 1 {
            self.location.x = self.location.x + 1;
        }
    }

    pub fn get_curr_reward(&self) -> f32 {
        self.curr_reward
    }

    pub fn set_curr_reward(&mut self, reward: f32) {
        self.curr_reward = reward;
    }

    pub fn get_location(&self) -> &Location {
        &self.location
    }

    pub fn get_output(&self) -> ResourceType {
        match self.agent_targets.iter().find(|target| target.station_type == StationType::DropOff) {
            Some(output) => output.resource.clone(),
            None => panic!("Agent has no output") // Should not happen ever
        }
    }

    pub fn get_current_target_locations(&self) -> Vec<Location> {
        self.agent_targets.iter().filter(|target| target.is_current_target && target.is_found).map(|target| target.location).collect()
    }


    fn take_resource(&mut self, resource: ResourceType) {
        self.agent_targets.iter_mut().for_each(|target| {
            if target.resource == resource && target.is_collected && target.station_type == StationType::DropOff {
                target.is_collected = false;
            }
        });
        // Successfully given output to agent
        // Now have to target inputs again
        self.reset_targets();
    }

    fn reset_targets(&mut self) {

        self.agent_targets.iter_mut().for_each(|target| {
            if target.station_type == StationType::PickUp {
                target.is_current_target = true;
            }
        });

    }

    // Attempts to combine current ingridients
    // Returns false is missing ingridients
    // Otherwise removes collected ingridients and adds output ingridient
    // Returns true
    fn try_combine(&mut self) -> bool {
        
        // Guard clause
        if !self.agent_targets.iter().all(|target| {
            (target.is_collected && target.station_type == StationType::PickUp) || (target.station_type == StationType::DropOff && !target.is_collected)
        }) { return false }

        // println!("Combining inputs...");

        self.agent_targets.iter_mut().for_each(|target| {
            if target.is_collected && target.station_type == StationType::PickUp {
                target.is_collected = false;
                target.is_current_target = false;
            }
            if target.station_type == StationType::DropOff {
                target.is_collected = true;
                target.is_current_target = true;
            }
        });

        true

    }
                

    pub fn get_self_observations(&self, width: u32, height: u32, max_targets: u32) -> Vec<f32> {
        let mut obs = Vec::new();

        // Is agent
        obs.push(0.0);

        // Location (0-1)
        obs.push(self.location.x as f32 / width as f32);
        obs.push(self.location.y as f32 / height as f32);

        // Info about targets
        let targets = self.agent_targets.iter().collect::<Vec<_>>();
        for i in 0..max_targets as usize {
            if i < targets.len() {
                obs.push(f32::from(&targets[i].resource));
                obs.push(targets[i].is_found as i32 as f32);
                obs.push(targets[i].is_collected as i32 as f32);
            } else {
                obs.push(0.0);
                obs.push(0.0);
                obs.push(0.0);
            }
        }

        obs

    }

    // Currently, only allow for progression
    // i.e. Do not overwrite items in inventory which sets
    // agent back
    pub fn interact_with_station(&mut self, station: &Station) -> f32 {

        const COMBINE_REWARD: f32 = 7.0;
        const BASIC_REWARD: f32 = 5.0;
        const FOUND_REWARD: f32 = 2.0;

        let mut successful_interaction = false;
        let mut dropped_off = false;
        let mut reward = 0.0;

        for target in self.agent_targets.iter_mut() {
            if target.station_type == *station.get_station_type() && target.resource == *station.get_resource() {
                if !target.is_found { // Hard code in memory of the agent
                    target.set_found(*station.get_location(), true);
                    reward += FOUND_REWARD;
                    if *station.get_station_type() == StationType::PickUp {
                        target.set_current_target();
                        // println!("FOUND: PICKUP")
                    } else {
                        // println!("FOUND: DROPOFF")
                    }
                }
                if target.is_current_target {
                    if target.station_type == StationType::PickUp {
                        target.collect();
                        // println!("COLLECTED: PICKUP")
                    } else {
                        target.dropoff();
                        dropped_off = true;
                        // println!("DROPPED: DROPOFF")
                    }
                    successful_interaction = true;
                }
            }
        }

        if successful_interaction {
            if self.try_combine() {
                reward += COMBINE_REWARD;
            } else {
                reward += BASIC_REWARD;
            }
        }
        if dropped_off {
            self.reset_targets();
        }

        reward

    }

    // Similar to interact_with_station
    // Only allow progression
    pub fn interact_with_agent(&mut self, agent: &mut Agent) -> f32 {

        const COMBINE_REWARD: f32 = 7.0;
        const BASIC_REWARD: f32 = 5.0;
        const FOUND_REWARD: f32 = 2.0;


        let mut successful_interaction = false;
        let mut reward = 0.0;

        for target in self.agent_targets.iter_mut() {
            if target.station_type == StationType::PickUp && target.resource == agent.get_output() {

                let other_agent_has_item = agent.agent_targets.iter().any(|t| 
                    t.station_type == StationType::DropOff
                    && t.resource == target.resource
                    && t.is_collected
                );

                if !target.is_found { // Hard code in agent memory, but note it is an agent - will move
                    target.set_found(agent.location, false);
                    reward += FOUND_REWARD;
                }
                if target.is_current_target && target.station_type == StationType::PickUp && other_agent_has_item {
                    agent.take_resource(target.resource);
                    successful_interaction = true;
                    target.collect();
                }
            }
        }

        if successful_interaction {
            if self.try_combine() {
                reward += COMBINE_REWARD;
            } else {
                reward += BASIC_REWARD;
            }
        }

        reward

    }

}

impl From<&Agent> for AgentState {
    fn from(agent: &Agent) -> Self {
        AgentState {
            id: agent.id,
            location: agent.location,
        }
    }
}