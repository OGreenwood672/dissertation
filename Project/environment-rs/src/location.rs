use std::ops::{Add, Sub};

use rand::Rng;
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Eq, Hash)]
pub struct Location {
    pub x: i32,
    pub y: i32,
}

impl Location {
    pub fn distance_squared(&self, other: Location) -> u64 {
        let dx = self.x.abs_diff(other.x) as u64;
        let dy = self.y.abs_diff(other.y) as u64;
        dx.pow(2) + dy.pow(2)
    }

    pub fn dot(self, other: Location) -> f32 {
        (self.x * other.x + self.y * other.y) as f32
    }

    pub fn magnitude(self) -> f32 {
        ((self.x.pow(2) + self.y.pow(2)) as f32).sqrt()
    }

    pub fn cosine_similarity(a: Location, b: Location, c: Location) -> f32 {
        let v1 = b - a;
        let v2 = c - a;

        let dot_product = v1.dot(v2);
        let mag_v1 = v1.magnitude();
        let mag_v2 = v2.magnitude();

        // Avoid division by zero
        if mag_v1 == 0.0 || mag_v2 == 0.0 {
            0.0
        } else {
            // (v1 . v2) / (||v1|| * ||v2||)
            dot_product / (mag_v1 * mag_v2)
        }
    }

}

impl Add for Location {
    type Output = Location;

    fn add(self, other: Location) -> Self::Output {
        Location {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Location {
    type Output = Location;

    fn sub(self, other: Location) -> Self::Output {
        Location {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}
impl Sub<&Location> for &Location {
    type Output = Location;

    fn sub(self, other: &Location) -> Self::Output {
        Location {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Serialize, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Layout {
    Random,
    #[serde(rename = "random-limited")]
    RandomLimited,
    Line,
    Circle,
}


fn get_random_location(max_x: u32, max_y: u32) -> Location {
    Location {
        x: rand::rng().random_range(0..max_x as i32),
        y: rand::rng().random_range(0..max_y as i32),
    }
}

// Gets x number of unique positions
// safe_radius = at least 2 * radius of entity
pub fn get_x_unique_random_locations(count: u32, max_x: u32, max_y: u32, safe_radius: f32, taken_points: &Vec<Location>) -> Vec<Location> {
    let mut locations: Vec<Location> = Vec::new();

    while locations.len() < count as usize {
        let location = get_random_location(max_x, max_y);
        let mut is_unique = true;

        for existing_location in &locations {
            if (location.x - existing_location.x).abs() <= safe_radius as i32 &&
               (location.y - existing_location.y).abs() <= safe_radius as i32 {
                is_unique = false;
                break;
            }
        }

        for existing_location in taken_points {
            if (location.x - existing_location.x).abs() <= safe_radius as i32 &&
                (location.y - existing_location.y).abs() <= safe_radius as i32 {
                is_unique = false;
                break;
            }
        }

        if is_unique {
            locations.push(location);
        }
    }

    locations
}

// Gets x number of unique positions
// safe_radius = at least 2 * radius of entity
pub fn get_x_unique_random_locations_with_limit(count: u32, max_x: u32, max_y: u32, safe_radius: f32, taken_points: &Vec<Location>) -> Vec<Location> {
    let mut locations: Vec<Location> = Vec::new();

    const X_RANGE: i32 = 5;
    const Y_RANGE: i32 = 5;

    let x_offset: i32 = (max_x as i32 - X_RANGE) / 2;
    let y_offset: i32 = (max_y as i32 - Y_RANGE) / 2;

    while locations.len() < count as usize {
        // let location = get_random_location(max_x, max_y);
        let location = get_random_location(X_RANGE as u32, Y_RANGE as u32) + Location { x: x_offset, y: y_offset };
        let mut is_unique = true;

        for existing_location in &locations {
            if (location.x - existing_location.x).abs() <= safe_radius as i32 &&
               (location.y - existing_location.y).abs() <= safe_radius as i32 {
                is_unique = false;
                break;
            }
        }

        for existing_location in taken_points {
            if (location.x - existing_location.x).abs() <= safe_radius as i32 &&
                (location.y - existing_location.y).abs() <= safe_radius as i32 {
                is_unique = false;
                break;
            }
        }

        if is_unique {
            locations.push(location);
        }
    }

    locations
}


fn get_equidistant_points_on_line(min_value: i32, max_value: i32, num_points: i32) -> Vec<i32> {
    let mut points = Vec::new();

    let step = (max_value - min_value) / num_points;

    for i in 1..(num_points+1) {
        points.push(min_value + i * step);
    }

    points
}


fn get_circle_points(center_x: u32, center_y: u32, radius: f32, num_points: u32) -> Vec<Location> {
    let mut points = Vec::new();

    for i in 0..num_points {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / num_points as f64;
        let x = center_x as f64 + radius as f64 * angle.cos();
        let y = center_y as f64 + radius as f64 * angle.sin();
        points.push(Location{ x: x as i32, y: y as i32 });
    }

    points
}

pub fn get_location(layout: Layout, width: u32, height: u32, entity_count: u32, safe_radius: f32) -> Vec<Location> {

    match layout {
        Layout::Circle => {
            get_circle_points(width / 2, height / 2, width as f32 / 2.0 * 0.8, entity_count)
        }
        Layout::Line => {
            get_equidistant_points_on_line(0, width as i32, entity_count as i32).into_iter().map(|x| Location{ x, y: (height / 2) as i32 }).collect()
        },
        Layout::Random => {
            get_x_unique_random_locations(entity_count, width, height, safe_radius, &Vec::new())
        },
        Layout::RandomLimited => {
            get_x_unique_random_locations_with_limit(entity_count, width, height, safe_radius, &Vec::new())
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_squared() {
        let a = Location { x: 0, y: 0 };
        let b = Location { x: 1, y: 1 };
        assert_eq!(a.distance_squared(b), 2);
    }

    #[test]
    fn test_dot() {
        let a = Location { x: 1, y: 2 };
        let b = Location { x: 3, y: 4 };
        assert_eq!(a.dot(b), 11.0);
    }

    #[test]
    fn test_magnitude() {
        let a = Location { x: 3, y: 4 };
        assert_eq!(a.magnitude(), 5.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Location { x: 1, y: 2 };
        let b = Location { x: 3, y: 4 };
        let c = Location { x: 5, y: 6 };
        let scale: f32 = 1000.0;
        assert_eq!(f32::round(Location::cosine_similarity(a, b, c) * scale) / scale, 1.0);
    }

    #[test]
    fn test_sub() {
        let a = Location { x: 1, y: 2 };
        let b = Location { x: 3, y: 4 };
        assert_eq!(a - b, Location { x: -2, y: -2 });
    }

    #[test]
    fn test_sub_ref() {
        let a = Location { x: 1, y: 2 };
        let b = Location { x: 3, y: 4 };
        assert_eq!(&a - &b, Location { x: -2, y: -2 });
    }

    #[test]
    fn test_add() {
        let a = Location { x: 1, y: 2 };
        let b = Location { x: 3, y: 4 };
        assert_eq!(a + b, Location { x: 4, y: 6 });
    }

    #[test]
    fn test_get_random_location() {
        let location = get_random_location(10, 1);
        assert!(location.x >= 0 && location.x < 10);
        assert!(location.y >= 0 && location.y < 1);
    }

    #[test]
    fn test_get_x_unique_random_locations() {
        let locations = get_x_unique_random_locations(3, 10, 10, 2.0, &Vec::new());
        assert_eq!(locations.len(), 3);
        for location in &locations {
            assert!(location.x >= 0 && location.x < 10);
            assert!(location.y >= 0 && location.y < 10);
        }
    }

    #[test]
    fn test_get_equidistant_points_on_line() {
        let points = get_equidistant_points_on_line(0, 10, 10);
        assert_eq!(points.len(), 10);
        assert_eq!(points[0], 1);
        assert_eq!(points[9], 10);
    }

    #[test]
    fn test_get_circle_points() {
        let points = get_circle_points(5, 5, 2.0, 2);
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].x, 7);
        assert_eq!(points[0].y, 5);
        assert_eq!(points[1].x, 3);
        assert_eq!(points[1].y, 5);
    }

        #[test]
    fn test_cosine_similarity_zero_magnitude() {
        let a = Location { x: 0, y: 0 };
        let b = Location { x: 0, y: 0 };
        let c = Location { x: 1, y: 0 };

        assert_eq!(Location::cosine_similarity(a, b, c), 0.0);
    }

    #[test]
    fn test_get_location_circle_layout() {
        let width = 10;
        let height = 10;
        let count = 4;
        let locs = get_location(Layout::Circle, width, height, count, 0.0);
        assert_eq!(locs.len(), count as usize);

        let center = Location { x: (width / 2) as i32, y: (height / 2) as i32 };
        for loc in &locs {
            let dx = (loc.x - center.x) as f32;
            let dy = (loc.y - center.y) as f32;
            let dist = (dx * dx + dy * dy).sqrt();
            assert!(dist <= (width as f32 / 2.0) * 0.9);
        }
    }

    #[test]
    fn test_get_location_line_layout() {
        let width = 10;
        let height = 10;
        let count = 5;
        let locs = get_location(Layout::Line, width, height, count, 0.0);
        assert_eq!(locs.len(), count as usize);

        let y_expected = (height / 2) as i32;
        for window in locs.windows(2) {
            assert_eq!(window[0].y, y_expected);
            assert_eq!(window[1].y, y_expected);
            assert!(window[0].x < window[1].x);
            assert!(window[0].x >= 0 && window[1].x <= width as i32);
        }
    }

    #[test]
    fn test_get_location_random_layout() {
        let width = 10;
        let height = 10;
        let count = 3;
        let locs = get_location(Layout::Random, width, height, count, 1.0);
        assert_eq!(locs.len(), count as usize);

        for loc in &locs {
            assert!(loc.x >= 0 && loc.x < width as i32);
            assert!(loc.y >= 0 && loc.y < height as i32);
        }
    }

    #[test]
    fn test_get_location_random_limited_layout() {
        let width = 20;
        let height = 20;
        let count = 3;
        let locs = get_location(Layout::RandomLimited, width, height, count, 1.0);
        assert_eq!(locs.len(), count as usize);

        let x_offset = (width as i32 - 5) / 2;
        let y_offset = (height as i32 - 5) / 2;

        for loc in &locs {
            assert!(loc.x >= x_offset && loc.x < x_offset + 5);
            assert!(loc.y >= y_offset && loc.y < y_offset + 5);
        }
    }

}