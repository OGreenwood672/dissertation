use serde::Deserialize;


#[derive(Debug, Deserialize, PartialEq, Eq, Copy, Clone)]
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
pub const RESOURCE_COUNT: usize = 9;

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

pub fn one_hot_vector_from_resource(resource: ResourceType) -> [f32; RESOURCE_COUNT] {
    let mut one_hot_vector = [0.0; RESOURCE_COUNT];
    one_hot_vector[f32::from(&resource) as usize - 1] = 1.0;
    one_hot_vector
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_vector_from_resource() {
        let resource = ResourceType::Burger;
        let one_hot_vector = one_hot_vector_from_resource(resource);
        assert_eq!(one_hot_vector[1], 1.0);
        assert_eq!(one_hot_vector[2], 0.0);
        assert_eq!(one_hot_vector[3], 0.0);
        assert_eq!(one_hot_vector[4], 0.0);
        assert_eq!(one_hot_vector[5], 0.0);
    }

    #[test]
    fn test_resource_to_f32() {
        let resource = ResourceType::Burger;
        assert_eq!(f32::from(&resource), 2.0);
        
    }
    
}