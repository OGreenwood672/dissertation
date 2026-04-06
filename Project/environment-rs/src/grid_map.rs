use std::{collections::{HashSet, VecDeque}, vec::Vec};

use crate::location::{Location};


// Used for neighbour detection
// Used for exploration reward

pub enum Entity {
    Station(usize),
    Agent(usize),
}

struct GridCell {
    agents: Vec<i32>,
    stations: Vec<i32>,
    visited: bool,
}

pub struct GridMap {
    width: u32,
    height: u32,
    cells: Vec<Vec<GridCell>>,
}

impl GridMap {

    pub fn new(width: u32, height: u32) -> Self {
        let mut cells = Vec::new();
        
        for _ in 0..height {
            let mut row = Vec::new();
            for _ in 0..width {
                row.push(GridCell {
                    agents: Vec::new(),
                    stations: Vec::new(),
                    visited: false,
                });
            }
            cells.push(row);
        }

        GridMap {
            width,
            height,
            cells,
        }
    }

    pub fn add_agent(&mut self, id: i32, location: Location) {
        self.cells[location.y as usize][location.x as usize].agents.push(id);
    }

    fn remove_agent(&mut self, id: i32, location: Location) {
        if let Some(pos) = self.cells[location.y as usize][location.x as usize].agents.iter().position(|&x| x == id) {
            self.cells[location.y as usize][location.x as usize].agents.swap_remove(pos);
        }
    }

    fn get_agents(&self, location: Location) -> &Vec<i32> {
        &self.cells[location.y as usize][location.x as usize].agents
    }

    pub fn has_location_been_visited(&self, location: Location) -> bool {
        self.cells[location.y as usize][location.x as usize].visited
    }

    pub fn set_location_visited(&mut self, location: Location) {
        self.cells[location.y as usize][location.x as usize].visited = true;
    }    

    pub fn add_station(&mut self, id: i32, location: Location) {
        self.cells[location.y as usize][location.x as usize].stations.push(id);
    }

    fn get_stations(&self, location: Location) -> &Vec<i32> {
        &self.cells[location.y as usize][location.x as usize].stations
    }

    pub fn reset(&mut self) {
        for row in &mut self.cells {
            for cell in row {
                cell.agents.clear();
                cell.stations.clear();
                cell.visited = false;
            }
        }
    }

    pub fn move_agent(&mut self, id: i32, old_location: Location, new_location: Location) {
        self.remove_agent(id, old_location);
        self.add_agent(id, new_location);
    }

    pub fn get_nearest_visible_entity(&self, id: i32, location: Location, max_radius: u32) -> Option<Entity> {

        let mut locations_searched = HashSet::new();
        let mut locations_to_search = VecDeque::new();

        locations_to_search.push_back(location);
        locations_searched.insert(location);

        // BFS
        while let Some(curr_location) = locations_to_search.pop_front() {

            for agent in self.get_agents(curr_location) {
                if *agent != id {
                    return Some(Entity::Agent(*agent as usize));
                }
            }

            let stations = self.get_stations(curr_location);
            if stations.len() > 0 {
                return Some(Entity::Station(stations[0] as usize));
            }

            for dx in -1..=1 {
                for dy in -1..=1 {
                    let candidate_location = Location { x: curr_location.x + dx, y: curr_location.y + dy };
                    if
                        // not self
                        (dx != 0 || dy != 0)
                        // not visited
                        && !locations_searched.contains(&Location { x: candidate_location.x, y: candidate_location.y })
                        // in radius
                        && (candidate_location - location).magnitude() <= max_radius as f32
                        // in world bounds
                        && candidate_location.x >= 0 && candidate_location.x < self.width as i32
                        && candidate_location.y >= 0 && candidate_location.y < self.height as i32
                    {
                        locations_to_search.push_back(candidate_location);
                        locations_searched.insert(candidate_location);

                    }

                }
            }

        }

        None

    }


}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_map() -> GridMap {
        GridMap::new(10, 10)
    }

    #[test]
    fn test_grid_init() {

        let map = setup_map();

        assert_eq!(map.width, 10);
        assert_eq!(map.height, 10);

        const LOCATION: Location = Location { x: 0, y: 0 };
        assert!(!map.has_location_been_visited(LOCATION));
    }

    #[test]
    fn test_agent_movement() {

        let mut map = setup_map();

        const ID: i32 = 1;
        const START: Location = Location { x: 2, y: 2 };
        const END: Location = Location { x: 3, y: 3 };
        
        // Add agent to start location
        map.add_agent(ID, START);
        assert_eq!(map.get_agents(START).len(), 1);
        
        // Move agent to end location
        map.move_agent(ID, START, END);
        assert_eq!(map.get_agents(START).len(), 0);
        assert_eq!(map.get_agents(END).len(), 1);
        assert_eq!(map.get_agents(END)[0], ID);
    }

    #[test]
    fn test_reset() {
        let mut map = setup_map();
        const LOCATION: Location = Location { x: 1, y: 1 };

        // Add agent and set location to visited
        map.add_agent(99, LOCATION);
        map.set_location_visited(LOCATION);
        
        map.reset();
        
        // Check reset works
        assert_eq!(map.get_agents(LOCATION).len(), 0);
        assert!(!map.has_location_been_visited(LOCATION));
    }

    #[test]
    fn test_nearest_entity_ignores_self() {

        let mut map = setup_map();

        let my_id = 1;
        let my_loc = Location { x: 5, y: 5 };
        
        map.add_agent(my_id, my_loc);
        
        let result = map.get_nearest_visible_entity(my_id, my_loc, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_nearest_entity() {

        let mut map = setup_map();

        const ID: i32 = 0;
        const LOCATION: Location = Location { x: 5, y: 5 };
        
        const AGENT_ID: i32 = 1;
        map.add_agent(AGENT_ID, Location { x: 5, y: 6 });
        
        const STATION_ID: i32 = 10;
        map.add_station(STATION_ID, Location { x: 5, y: 8 });
        
        // Check agent is found nearest
        match map.get_nearest_visible_entity(ID, LOCATION, 5) {
            Some(Entity::Agent(id)) => assert_eq!(id, AGENT_ID as usize),
            _ => panic!("Expected to find agent 1"),
        }
    }

    #[test]
    fn test_max_radius() {

        let mut map = setup_map();

        const ID: i32 = 0;
        const LOCATION: Location = Location { x: 0, y: 0 };
        
        map.add_agent(1, Location { x: 5, y: 5 });

        assert!(map.get_nearest_visible_entity(ID, LOCATION, 2).is_none());
    }
}