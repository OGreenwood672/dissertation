use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use crate::location::Location;
use crate::resource::{ResourceType, one_hot_vector_from_resource};

#[derive(Debug, Deserialize, PartialEq, Copy, Clone, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum StationType {
    PickUp,
    DropOff,
}

impl From<&StationType> for f32 {
    fn from(station_type: &StationType) -> Self {
        match station_type {
            StationType::PickUp => 0.0,
            StationType::DropOff => 1.0,
        }
    }
}


pub struct Station {
    id: i32,
    location: Location,
    station_type: StationType,
    resource: ResourceType,
    is_found: AtomicBool,
}

#[derive(Serialize)]
pub struct StationState {
    id: i32,
    location: Location
}

impl Station {
    pub fn new(id: i32, location: Location, station_type: StationType, resource: ResourceType) -> Self {
        Station { id, location, station_type, resource, is_found: AtomicBool::new(false) }
    }

    pub fn get_self_observations(&self, width: u32, height: u32) -> Vec<f32> {
        let mut obs = Vec::new();

        // Is station
        obs.push(1.0);

        // Location (0-1)
        obs.push(self.location.x as f32 / width as f32);
        obs.push(self.location.y as f32 / height as f32);

        // Pickup || Dropoff
        obs.push(f32::from(&self.station_type));

        // Resource
        obs.extend(one_hot_vector_from_resource(self.resource));


        obs
    }

    pub fn get_id(&self) -> i32 {
        self.id
    }

    pub fn get_location(&self) -> &Location {
        &self.location
    }

    // For testing
    pub fn set_location(&mut self, location: Location) {
        self.location = location;
    }

    pub fn get_station_type(&self) -> &StationType {
        &self.station_type
    }
    
    pub fn get_resource(&self) -> &ResourceType {
        &self.resource
    }

    pub fn get_is_found(&self) -> bool {
        self.is_found.load(Ordering::Relaxed)
    }

    pub fn set_found(&self) {
        self.is_found.store(true, Ordering::Relaxed);
    }
}

impl From<&Station> for StationState {
    fn from(station: &Station) -> Self {
        StationState {
            id: station.id,
            location: station.location,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_station_type_to_f32() {
        let station_type = StationType::PickUp;
        assert_eq!(f32::from(&station_type), 0.0);
    }

    #[test]
    fn test_station_new() {
        let station = Station::new(1, Location { x: 0, y: 0 }, StationType::PickUp, ResourceType::Burger);
        assert_eq!(station.id, 1);
        assert_eq!(station.location, Location { x: 0, y: 0 });
        assert_eq!(station.station_type, StationType::PickUp);
        assert_eq!(station.resource, ResourceType::Burger);
        assert!(!station.is_found.load(Ordering::Relaxed));
    }

    #[test]
    fn test_station_get_self_observations() {
        let station = Station::new(1, Location { x: 5, y: 5 }, StationType::PickUp, ResourceType::Burger);
        let obs = station.get_self_observations(10, 10);
        assert_eq!(obs.len(), 13);
        assert_eq!(obs[0], 1.0);
        assert_eq!(obs[1], 0.5);
        assert_eq!(obs[2], 0.5);
        assert_eq!(obs[3], 0.0);
        assert_eq!(obs[4], 0.0);
        assert_eq!(obs[5], 1.0);
        assert_eq!(obs[6], 0.0);
        assert_eq!(obs[7], 0.0);
        assert_eq!(obs[8], 0.0);
        assert_eq!(obs[9], 0.0);
        assert_eq!(obs[10], 0.0);
        assert_eq!(obs[11], 0.0);
        assert_eq!(obs[12], 0.0);
    }

    #[test]
    fn test_station_get_id() {
        let station = Station::new(1, Location { x: 0, y: 0 }, StationType::PickUp, ResourceType::Burger);
        assert_eq!(station.get_id(), 1);
    }

    #[test]
    fn test_station_get_location() {
        let station = Station::new(1, Location { x: 0, y: 0 }, StationType::DropOff, ResourceType::Burger);
        assert_eq!(station.get_location(), &Location { x: 0, y: 0 });
    }

    #[test]
    fn test_station_get_station_type() {
        let station = Station::new(1, Location { x: 0, y: 0 }, StationType::DropOff, ResourceType::Burger);
        assert_eq!(station.get_station_type(), &StationType::DropOff);
    }

    #[test]
    fn test_station_get_resource() {
        let station = Station::new(1, Location { x: 0, y: 0 }, StationType::PickUp, ResourceType::Burger);
        assert_eq!(station.get_resource(), &ResourceType::Burger);
    }

    #[test]
    fn test_station_get_is_found() {
        let station = Station::new(1, Location { x: 0, y: 0 }, StationType::PickUp, ResourceType::Burger);
        assert!(!station.get_is_found());
    }

    #[test]
    fn test_station_set_found() {
        let station = Station::new(1, Location { x: 0, y: 0 }, StationType::PickUp, ResourceType::Burger);
        station.set_found();
        assert!(station.get_is_found());
    }

    #[test]
    fn test_station_from_station_state() {
        let station = Station::new(1, Location { x: 0, y: 0 }, StationType::PickUp, ResourceType::Burger);
        let station_state = StationState::from(&station);
        assert_eq!(station_state.id, 1);
        assert_eq!(station_state.location, Location { x: 0, y: 0 });
    }

    #[test]
    fn test_station_set_location() {
        let mut station = Station::new(1, Location { x: 0, y: 0 }, StationType::PickUp, ResourceType::Burger);
        station.set_location(Location { x: 1, y: 1 });
        assert_eq!(station.location, Location { x: 1, y: 1 });
    }

    
}