use std::vec::Vec;

use crate::agent::{Agent};
use crate::station::Station;
use crate::config::{AgentConfig, Config, StationConfig};
use crate::location::{get_location};


pub struct World {
    width: u32,
    height: u32,
    agents: Vec<Agent>,
    stations: Vec<Station>
}

impl World {
    pub fn new(config: Config) -> Self {

        let width = config.width;
        let height = config.height;

        let mut station_locations = get_location(
            config.station_layout,
            width,
            height,
            config.stations.len() as u32,
            config.station_size * 3
        ).into_iter();

        let stations: Vec<Station> = config.stations.into_iter().map(|station: StationConfig| Station::new(
            station.id as i32,
            station_locations.next().unwrap(),
            station.station_type,
            station.resource,
        )).collect();

        
        let mut initial_agent_locations = get_location(
            config.initial_agent_layout,
            width, height,
            config.agents.len() as u32,
            config.agent_size * 3
        ).into_iter();

        let agents: Vec<Agent> = config.agents.into_iter().map(|agent: AgentConfig| Agent::new(
            agent.id as i32,
            initial_agent_locations.next().unwrap(),
            agent.inputs,
            agent.output,
        )).collect();

        World {
            width,
            height,
            agents,
            stations
        }
    }
}