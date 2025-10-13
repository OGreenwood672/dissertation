use crate::location::Location;
use crate::resource::ResourceType;

pub struct Agent {
    pub id: i32,
    location: Location,
    inventory: Vec<ResourceType>,
    input: Vec<ResourceType>,
    output: ResourceType,
}
// actions: move up, down, left, right, take, drop, 

impl Agent {
    /// Creates a new Agent instance.
    pub fn new(id: i32, location: Location, input: Vec<ResourceType>, output: ResourceType) -> Self {
        Agent {
            id,
            location,
            inventory: Vec::new(),
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
}