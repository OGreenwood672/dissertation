use std::convert::From;
use std::vec::Vec;
use serde::Serialize;

use crate::location::Location;
use crate::resource::ResourceType;
use crate::station::{Station, StationType};

#[derive(Serialize)]
pub struct AgentState {
    pub id: i32,
    pub location: Location,
}

#[derive(Clone)]
pub struct Agent {
    pub id: i32,
    pub location: Location,
    inventory: Option<ResourceType>,
    input: Vec<ResourceType>,
    output: ResourceType,
}

// actions: move up, down, left, right, interact, 
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
    pub fn new(id: i32, location: Location, input: Vec<ResourceType>, output: ResourceType) -> Self {

        assert!(input.len() <= 2);

        Agent {
            id,
            location,
            inventory: None,
            input,
            output
        }
    }

    /// Moves the agent one unit north.
    pub fn move_north(&mut self) {
        self.location.y = self.location.y.saturating_sub(1);
    }

    /// Moves the agent one unit south.
    pub fn move_south(&mut self) {
        self.location.y = self.location.y.saturating_add(1);
    }

    /// Moves the agent one unit west.
    pub fn move_west(&mut self) {
        self.location.x = self.location.x.saturating_sub(1);
    }

    /// Moves the agent one unit east.
    pub fn move_east(&mut self) {
        self.location.x = self.location.x.saturating_add(1);
    }

    pub fn get_self_observations(&self, width: u32, height: u32) -> Vec<f32> {
        let mut obs = Vec::new();

        // Is agent
        obs.push(0.0);

        // Location (0-1)
        obs.push(self.location.x as f32 / width as f32);
        obs.push(self.location.y as f32 / height as f32);

        // Input
        obs.push(f32::from(&self.input[0]));
        if self.input.len() > 1 {
            obs.push(f32::from(&self.input[1]));
        } else {
            obs.push(0.0);
        }

        // Output
        obs.push(f32::from(&self.output));

        // Inventory
        if let Some(item) = self.inventory {
            obs.push(f32::from(&item));
        } else {
            obs.push(0.0);
        }

        obs

    }

    // Currently, only allow for progression
    // i.e. Do not overwrite items in inventory which sets
    // agent back
    pub fn interact_with_station(&mut self, station: &Station) -> f32 {

        const COMBINE_REWARD: f32 = 10.0;
        const PICKUP_REWARD: f32 = 2.0;
        const DROP_OFF_REWARD: f32 = 25.0;

        if station.station_type == StationType::PickUp {
            if self.input.len() == 2 {
                if let Some(inventory) = self.inventory { // item in inventory
                    if self.input.contains(&station.resource) && self.input.contains(&inventory) && inventory != station.resource {
                        // inputs are our current inventory and whats at the station
                        self.inventory = Some(self.output);
                        COMBINE_REWARD
                    } else {
                        0.0
                    }
                } else if self.input.contains(&station.resource) { // inventory empty
                    self.inventory = Some(station.resource);
                    PICKUP_REWARD
                } else {
                    0.0
                }
            } else if self.input.len() == 1 {
                if self.input.contains(&station.resource) { // Items needed is at this station
                    self.inventory = Some(self.output);
                    COMBINE_REWARD
                } else {
                    0.0
                }
            } else {
                0.0
            }

        // Otherwise drop off item if at correct station
        } else {
            if let Some(inventory) = self.inventory { // item in inventory
                if inventory == station.resource { // item needed at station is the one we have
                    self.inventory = None;
                    DROP_OFF_REWARD
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }

    }

    // Similar to interact_with_station
    // Only allow progression
    pub fn interact_with_agent(&mut self, agent: &mut Agent) -> f32 {

        const COLLECT_REWARD: f32 = 10.0;

        if let Some(other_inventory) = agent.inventory { // Other's inventory
            if let Some(inventory) = self.inventory { // Our inventory

                if self.input.contains(&inventory) && inventory != other_inventory { // We don't have but need
                    agent.inventory = None;
                    self.inventory = Some(other_inventory);
                    COLLECT_REWARD
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        }

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