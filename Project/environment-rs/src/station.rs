use serde::{Deserialize, Serialize};

use crate::location::Location;
use crate::resource::ResourceType;

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
}

#[derive(Serialize)]
pub struct StationState {
    id: i32,
    location: Location
}

impl Station {
    pub fn new(id: i32, location: Location, station_type: StationType, resource: ResourceType) -> Self {
        Station { id, location, station_type, resource }
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
        obs.push(f32::from(&self.resource));


        obs
    }

    pub fn get_location(&self) -> &Location {
        &self.location
    }

    pub fn get_station_type(&self) -> &StationType {
        &self.station_type
    }
    
    pub fn get_resource(&self) -> &ResourceType {
        &self.resource
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