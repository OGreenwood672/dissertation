use serde::Deserialize;

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

impl Station {
    pub fn new(id: i32, location: Location, station_type: StationType, resource: ResourceType) -> Self {
        Station { id, location, station_type, resource }
    }
}