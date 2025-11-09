use serde::Deserialize;


#[derive(Debug, Deserialize, PartialEq, Copy, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ResourceType {
    PlainBurger,
    Burger,
    Tomato,
    Lettuce,
    Patty,
    SlicedTomato,
    ChoppedLettuce,
    Bun,
    Salad
}

impl From<&ResourceType> for f32 {
    fn from(resource: &ResourceType) -> Self {
        match resource {
            ResourceType::PlainBurger => 1.0,
            ResourceType::Burger => 2.0,
            ResourceType::Tomato => 3.0,
            ResourceType::Lettuce => 4.0,
            ResourceType::Patty => 5.0,
            ResourceType::SlicedTomato => 6.0,
            ResourceType::ChoppedLettuce => 7.0,
            ResourceType::Bun => 8.0,
            ResourceType::Salad => 9.0,
        }
    }
}
