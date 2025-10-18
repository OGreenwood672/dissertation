use serde::{Deserialize, Serialize};

use crate::location::Location;
use crate::resource::ResourceType;

#[derive(Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum StationType {
    PickUp,
    DropOff,
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
}

impl From<&Station> for StationState {
    fn from(station: &Station) -> Self {
        StationState {
            id: station.id,
            location: station.location,
        }
    }
}