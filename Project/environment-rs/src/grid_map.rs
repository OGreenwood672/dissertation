use std::{vec::Vec};

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
    // width: u32,
    // height: u32,
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
            // width,
            // height,
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

    // fn get_agents(&self, location: Location) -> &Vec<i32> {
    //     &self.cells[location.y as usize][location.x as usize].agents
    // }

    pub fn has_location_been_visited(&self, location: Location) -> bool {
        self.cells[location.y as usize][location.x as usize].visited
    }

    pub fn set_location_visited(&mut self, location: Location) {
        self.cells[location.y as usize][location.x as usize].visited = true;
    }    

    pub fn add_station(&mut self, id: i32, location: Location) {
        self.cells[location.y as usize][location.x as usize].stations.push(id);
    }

    // fn get_stations(&self, location: Location) -> &Vec<i32> {
    //     &self.cells[location.y as usize][location.x as usize].stations
    // }


    pub fn move_agent(&mut self, id: i32, old_location: Location, new_location: Location) {
        self.remove_agent(id, old_location);
        self.add_agent(id, new_location);
    }

    // pub fn get_nearest_visible_entity(&self, id: i32, location: Location, max_radius: u32) -> Option<Entity> {

    //     let mut locations_searched = HashSet::new();
    //     let mut locations_to_search = VecDeque::new();

    //     locations_to_search.push_back(location);
    //     locations_searched.insert(location);

    //     // BFS
    //     while let Some(curr_location) = locations_to_search.pop_front() {

    //         for agent in self.get_agents(curr_location) {
    //             if *agent != id {
    //                 return Some(Entity::Agent(*agent as usize));
    //             }
    //         }

    //         let stations = self.get_stations(curr_location);
    //         if stations.len() > 0 {
    //             return Some(Entity::Station(stations[0] as usize));
    //         }

    //         for dx in -1..=1 {
    //             for dy in -1..=1 {
    //                 let candidate_location = Location { x: curr_location.x + dx, y: curr_location.y + dy };
    //                 if
    //                     // not self
    //                     (dx != 0 || dy != 0)
    //                     // not visited
    //                     && !locations_searched.contains(&Location { x: candidate_location.x, y: candidate_location.y })
    //                     // in radius
    //                     && (candidate_location - location).magnitude() <= max_radius as f32
    //                     // in world bounds
    //                     && candidate_location.x >= 0 && candidate_location.x < self.width as i32
    //                     && candidate_location.y >= 0 && candidate_location.y < self.height as i32
    //                 {
    //                     locations_to_search.push_back(candidate_location);
    //                     locations_searched.insert(candidate_location);

    //                 }

    //             }
    //         }

    //     }

    //     None

    // }


}