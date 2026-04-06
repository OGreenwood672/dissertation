use std::convert::From;
use std::vec::Vec;
use serde::Serialize;

use crate::location::Location;
use crate::resource::{RESOURCE_COUNT, ResourceType, one_hot_vector_from_resource};
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
    collection_count: i32,
}

// actions: move up, down, left, right, interact, 
pub const ACTION_COUNT: u32 = 5;
#[derive(PartialEq, Debug)]
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

impl From<Action> for i32 {
    fn from(item: Action) -> Self {
        match item {
            Action::MoveUp => 0,
            Action::MoveDown => 1,
            Action::MoveLeft => 2,
            Action::MoveRight => 3,
            Action::Interact => 4,
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

        let mut agent_targets: Vec<AgentTarget> = input.into_iter().map(|input| {
            AgentTarget::new(StationType::PickUp, input)
        }).collect();
        agent_targets.push(AgentTarget::new(StationType::DropOff, output));

        let mut agent = Agent {
            id,
            location,
            curr_reward: 0.0,
            agent_targets,
            collection_count: 0,
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

    pub fn add_curr_reward(&mut self, reward: f32) {
        self.curr_reward += reward;
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

    pub fn get_inputs(&self) -> Vec<ResourceType> {
        self.agent_targets.iter().filter(|target| target.station_type == StationType::PickUp).map(|target| target.resource.clone()).collect()
    }

    pub fn get_curr_output(&self) -> Option<ResourceType> {
        self.agent_targets.iter().find(|target| target.station_type == StationType::DropOff && target.is_collected).map(|target| target.resource.clone())
    }


    pub fn get_current_targets(&self) -> Vec<&AgentTarget> {
        self.agent_targets.iter().filter(|target| target.is_current_target && target.is_found).map(|target| target).collect()
    }

    pub fn is_needed(&self, resource: ResourceType, station_type: StationType) -> bool {
        self.agent_targets.iter().any(|target| {
            target.resource == resource && target.station_type == station_type && target.is_current_target
        })
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
        self.add_curr_reward(0.7);
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
                obs.extend(one_hot_vector_from_resource(targets[i].resource));
                obs.push(targets[i].is_found as i32 as f32);
                obs.push(targets[i].is_collected as i32 as f32);
            } else {
                obs.extend(vec![0.0; RESOURCE_COUNT]);
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

        const COMBINE_REWARD: f32 = 1.0;
        const BASIC_REWARD: f32 = 0.7;
        const FOUND_REWARD: f32 = 0.5;

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
            self.collection_count += 1;
        }

        if dropped_off {
            self.reset_targets();
        }

        reward

    }

    // Similar to interact_with_station
    // Only allow progression
    pub fn interact_with_agent(&mut self, agent: &mut Agent) -> f32 {

        const COMBINE_REWARD: f32 = 1.0;
        const BASIC_REWARD: f32 = 0.7;
        const FOUND_REWARD: f32 = 0.5;


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
                    reward += FOUND_REWARD;
                }
                target.set_found(agent.location, false);

                if target.is_current_target && target.station_type == StationType::PickUp && other_agent_has_item {
                    agent.take_resource(target.resource);
                    successful_interaction = true;
                    target.collect();
                }
            } else if target.station_type == StationType::DropOff && agent.get_inputs().contains(&target.resource) {

                if !target.is_found { // Hard code in agent memory, but note it is an agent - will move
                    reward += FOUND_REWARD;
                }
                target.set_found(agent.location, false);

            }
                
        }

        if successful_interaction {
            if self.try_combine() {
                reward += COMBINE_REWARD;
            } else {
                reward += BASIC_REWARD;
            }
            self.collection_count += 1;
        }

        reward

    }

    pub fn get_inventory_reward(&self) -> f32 {
        // let mut reward = 0.0;
        // for target in &self.agent_targets {
        //     if target.is_collected {
        //         reward += 0.005; 
        //     }
        // }
        // reward

        self.collection_count as f32 * 0.004
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_new() {
        let location = Location { x: 0, y: 0 };
        let input = vec![ResourceType::Burger];
        let output = ResourceType::Burger;
        let agent = Agent::new(1, location, input, output);
        assert_eq!(agent.id, 1);
        assert_eq!(agent.location, location);
        assert_eq!(agent.curr_reward, 0.0);
        assert_eq!(agent.agent_targets.len(), 2);
        assert_eq!(agent.agent_targets[0].resource, ResourceType::Burger);
        assert_eq!(agent.agent_targets[0].station_type, StationType::PickUp);
        assert_eq!(agent.agent_targets[0].is_found, false);
        assert_eq!(agent.agent_targets[0].is_current_target, true);
        assert_eq!(agent.agent_targets[0].is_collected, false);
    }

    #[test]
    fn test_agent_move_north() {
        let mut agent = Agent::new(1, Location { x: 0, y: 2 }, vec![], ResourceType::Burger);
        agent.move_north();
        assert_eq!(agent.location, Location { x: 0, y: 1 });
    }

    #[test]
    fn test_agent_move_invalid_north() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![], ResourceType::Burger);
        agent.move_north();
        assert_eq!(agent.location, Location { x: 0, y: 0 });
    }

    #[test]
    fn test_agent_move_south() {
        let mut agent = Agent::new(1, Location { x: 0, y: 2 }, vec![], ResourceType::Burger);
        agent.move_south(10);
        assert_eq!(agent.location, Location { x: 0, y: 3 });
    }

    #[test]
    fn test_agent_move_invalid_south() {
        let mut agent = Agent::new(1, Location { x: 0, y: 9 }, vec![], ResourceType::Burger);
        agent.move_south(10);
        assert_eq!(agent.location, Location { x: 0, y: 9 });
    }

    #[test]
    fn test_agent_move_west() {
        let mut agent = Agent::new(1, Location { x: 2, y: 0 }, vec![], ResourceType::Burger);
        agent.move_west();
        assert_eq!(agent.location, Location { x: 1, y: 0 });
    }

    #[test]
    fn test_agent_move_invalid_west() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![], ResourceType::Burger);
        agent.move_west();
        assert_eq!(agent.location, Location { x: 0, y: 0 });
    }

    #[test]
    fn test_agent_move_east() {
        let mut agent = Agent::new(1, Location { x: 2, y: 0 }, vec![], ResourceType::Burger);
        agent.move_east(10);
        assert_eq!(agent.location, Location { x: 3, y: 0 });
    }

    #[test]
    fn test_agent_move_invalid_east() {
        let mut agent = Agent::new(1, Location { x: 9, y: 0 }, vec![], ResourceType::Burger);
        agent.move_east(10);
        assert_eq!(agent.location, Location { x: 9, y: 0 });
    }

    #[test]
    fn test_agent_set_curr_reward() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![], ResourceType::Burger);
        agent.set_curr_reward(1.0);
        assert_eq!(agent.get_curr_reward(), 1.0);
    }

    #[test]
    fn test_agent_add_curr_reward() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![], ResourceType::Burger);
        agent.add_curr_reward(1.0);
        assert_eq!(agent.get_curr_reward(), 1.0);
    }

    #[test]
    fn test_agent_get_location() {
        let agent = Agent::new(1, Location { x: 0, y: 0 }, vec![], ResourceType::Burger);
        assert_eq!(agent.get_location(), &Location { x: 0, y: 0 });
    }

    #[test]
    fn test_agent_get_inputs() {
        let agent = Agent::new(1, Location { x: 0, y: 0 }, vec![ResourceType::Burger], ResourceType::Burger);
        assert_eq!(agent.get_inputs(), vec![ResourceType::Burger]);
    }

    #[test]
    fn test_agent_get_output() {
        let agent = Agent::new(1, Location { x: 0, y: 0 }, vec![], ResourceType::Burger);
        assert_eq!(agent.get_output(), ResourceType::Burger);
    }

    #[test]
    fn test_agent_get_curr_target() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![], ResourceType::Burger);
        // Set target
        agent.agent_targets[0].set_found(Location { x: 0, y: 0 }, true);
        agent.agent_targets[0].set_current_target();

        assert_eq!(agent.get_current_targets().len(), 1);
    }

    #[test]
    fn test_agent_is_needed() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0}, vec![], ResourceType::Burger);
        agent.agent_targets[0].set_found(Location { x: 0, y: 0 }, true);
        agent.agent_targets[0].set_current_target();
        assert!(agent.is_needed(ResourceType::Burger, StationType::DropOff));
    }

    #[test]
    fn test_agent_take_resource() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![], ResourceType::Burger);

        agent.agent_targets[0].set_found(Location { x: 0, y: 0 }, true);
        agent.agent_targets[0].is_collected = true;
        agent.take_resource(ResourceType::Burger);

        assert_eq!(agent.agent_targets[0].is_collected, false);
        assert_eq!(agent.get_curr_reward(), 0.7);
    }

    #[test]
    fn test_agent_reset_targets() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![ResourceType::Bun, ResourceType::Patty], ResourceType::Burger);
        agent.agent_targets[0].collect();
        agent.agent_targets[0].set_found(Location { x: 0, y: 0 }, true);
        agent.agent_targets[1].collect();
        agent.agent_targets[1].set_found(Location { x: 0, y: 0 }, true);
        assert_eq!(agent.get_current_targets().len(), 0);
        agent.reset_targets();
        assert_eq!(agent.get_current_targets().len(), 2);
    }

    #[test]
    fn test_agent_try_combine() {

        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![ResourceType::Bun, ResourceType::Patty], ResourceType::Burger);
        agent.agent_targets[0].set_found(Location { x: 0, y: 0 }, true);
        agent.agent_targets[1].set_found(Location { x: 0, y: 0 }, true);
        agent.agent_targets[0].collect();
        agent.agent_targets[1].collect();
        agent.try_combine();
        assert!(agent.agent_targets[2].is_collected);
    
    }

    #[test]
    fn test_agent_get_self_observations() {
        let agent = Agent::new(1, Location { x: 5, y: 5 }, vec![], ResourceType::Burger);
        let max_targets = 3;
        let obs = agent.get_self_observations(10, 10, max_targets);
        assert_eq!(obs.len() as u32, 3 + max_targets * (RESOURCE_COUNT as u32 + 2));
        assert_eq!(obs[0], 0.0);
        assert_eq!(obs[1], 0.5);
        assert_eq!(obs[2], 0.5);    
    }

    #[test]
    fn test_agent_interact_with_station() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![ResourceType::Bun], ResourceType::Burger);
        let station = Station::new(1, Location { x: 0, y: 0 }, StationType::PickUp, ResourceType::Bun);
        let reward = agent.interact_with_station(&station);
        assert_eq!(reward, 1.5);
        assert_eq!(agent.agent_targets[0].is_found, true);
        assert_eq!(agent.agent_targets[0].is_current_target, false);
        assert_eq!(agent.agent_targets[0].is_collected, false);
    }

    #[test]
    fn test_agent_interact_with_agent() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![ResourceType::Burger], ResourceType::Burger);
        let mut other_agent = Agent::new(2, Location { x: 0, y: 0 }, vec![], ResourceType::Burger);
        agent.agent_targets[0].set_found(Location { x: 0, y: 0 }, true);
        other_agent.agent_targets[0].set_found(Location { x: 0, y: 0 }, true);
        other_agent.agent_targets[0].collect();
        let reward = agent.interact_with_agent(&mut other_agent);
        assert_eq!(reward, 1.0);
        assert_eq!(agent.agent_targets[0].is_found, true);
        assert_eq!(agent.agent_targets[0].is_current_target, false);
        assert_eq!(agent.agent_targets[0].is_collected, false);
    }

    #[test]
    fn test_agent_get_inventory_reward() {
        let mut agent = Agent::new(1, Location { x: 0, y: 0 }, vec![ResourceType::Bun, ResourceType::Patty], ResourceType::Burger);
        let mut other_agent = Agent::new(2, Location { x: 0, y: 0 }, vec![], ResourceType::Bun);
        other_agent.agent_targets[0].set_found(Location { x: 0, y: 0 }, true);
        other_agent.agent_targets[0].collect();
        agent.interact_with_agent(&mut other_agent);
        let reward = agent.get_inventory_reward();
        assert_eq!(reward, 0.004);
    }

    #[test]
    fn test_agent_state_from_agent() {
        let agent = Agent::new(1, Location { x: 0, y: 0 }, vec![], ResourceType::Burger);
        let agent_state = AgentState::from(&agent);
        assert_eq!(agent_state.id, 1);
        assert_eq!(agent_state.location, Location { x: 0, y: 0 });
    }

    #[test]
    fn test_agent_target_new() {
        let target = AgentTarget::new(StationType::PickUp, ResourceType::Burger);
        assert_eq!(target.is_station, false);
        assert_eq!(target.station_type, StationType::PickUp);
        assert_eq!(target.resource, ResourceType::Burger);
        assert_eq!(target.location, Location { x: -1, y: -1 });
        assert_eq!(target.is_found, false);
        assert_eq!(target.is_current_target, false);
        assert_eq!(target.is_collected, false);
    }

    #[test]
    fn test_agent_target_collect() {
        let mut target = AgentTarget::new(StationType::PickUp, ResourceType::Burger);
        target.collect();
        assert_eq!(target.is_current_target, false);
        assert_eq!(target.is_collected, true);
    }

    #[test]
    fn test_agent_target_dropoff() {
        let mut target = AgentTarget::new(StationType::PickUp, ResourceType::Burger);
        target.set_current_target();
        target.dropoff();
        assert_eq!(target.is_current_target, false);
        assert_eq!(target.is_collected, false);
    }

    #[test]
    fn test_agent_target_set_found() {
        let mut target = AgentTarget::new(StationType::PickUp, ResourceType::Burger);
        target.set_found(Location { x: 0, y: 0 }, true);
        assert_eq!(target.is_found, true);
        assert_eq!(target.location, Location { x: 0, y: 0 });
        assert_eq!(target.is_station, true);
    }

    #[test]
    fn test_agent_target_set_current_target() {
        let mut target = AgentTarget::new(StationType::PickUp, ResourceType::Burger);
        target.set_current_target();
        assert_eq!(target.is_current_target, true);
    }

    #[test]
    fn test_action_from_i32() {
        let action = Action::from(0);
        assert_eq!(action, Action::MoveUp);
        let action = Action::from(1);
        assert_eq!(action, Action::MoveDown);
        let action = Action::from(2);
        assert_eq!(action, Action::MoveLeft);
        let action = Action::from(3);
        assert_eq!(action, Action::MoveRight);
        let action = Action::from(4);
        assert_eq!(action, Action::Interact);
    }

    #[test]
    fn test_action_to_i32() {
        let action = Action::MoveUp;
        assert_eq!(Into::<i32>::into(action), 0);
        let action = Action::MoveDown;
        assert_eq!(Into::<i32>::into(action), 1);
        let action = Action::MoveLeft;
        assert_eq!(Into::<i32>::into(action), 2);
        let action = Action::MoveRight;
        assert_eq!(Into::<i32>::into(action), 3);
        let action = Action::Interact;
        assert_eq!(Into::<i32>::into(action), 4);
    }

}