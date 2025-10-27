use serde::Deserialize;


#[derive(Debug, Deserialize, PartialEq)]
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
