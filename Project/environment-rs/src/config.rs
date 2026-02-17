use serde::Deserialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::location::Layout;
use crate::resource::ResourceType;
use crate::station::StationType;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub arena_width: u32,
    pub arena_height: u32,
    pub headless: bool,
    pub websocket_url: String,
    pub websocket_path: String,
    pub agent_agent_visibility: f32,
    pub agent_station_visibility: f32,
    pub initial_agent_layout: Layout,
    pub station_layout: Layout,
    pub agents: Vec<AgentConfig>,
    pub stations: Vec<StationConfig>,
}

/// Represents an agent that can process resources.
#[derive(Debug, Deserialize, Clone)]
pub struct AgentConfig {
    pub id: u32,
    pub inputs: Vec<ResourceType>,
    pub output: ResourceType,
}

/// Represents a station for picking up or dropping off resources.
#[derive(Debug, Deserialize, Clone)]
pub struct StationConfig {
    pub id: u32,
    #[serde(rename = "type")]
    pub station_type: StationType,
    pub resource: ResourceType,
}

/// Loads, parses, and returns the application configuration from a YAML file.
/// This function will return an error if the file cannot be opened, read,
/// or if the YAML content cannot be parsed into the `Config` struct.
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<Config, Box<dyn std::error::Error>> {
    // Open the file
    let mut file = File::open(path)?;

    // Read the file content into a string
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    // Parse the YAML string into the Config struct
    let config: Config = serde_yaml::from_str(&contents)?;

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_config() {
        let yaml_content = r#"
            arena_width: 400
            arena_height: 400
            headless: true
            websocket_url: ws://127.0.0.1:3000
            websocket_path: ws
            agent_agent_visibility: 10.0
            agent_station_visibility: 10.0
            initial_agent_layout: random
            station_layout: circle
            agents:
              - id: 1
                inputs: [lettuce]
                output: choppedlettuce
            stations:
              - id: 1
                type: pickup
                resource: bun
        "#;
        let config: Config = serde_yaml::from_str(yaml_content).expect("Failed to parse test YAML");
        assert_eq!(config.arena_width, 400);
        assert_eq!(config.agents.len(), 1);
        assert_eq!(config.stations.len(), 1);
        assert!(config.agents[0].inputs.contains(&ResourceType::Lettuce));
        assert_eq!(config.stations[0].resource, ResourceType::Bun);
    }
}