use std::ops::Sub;

use rand::Rng;
use serde::{Deserialize, Serialize};


#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
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

impl Sub for Location {
    type Output = Location;

    fn sub(self, other: Location) -> Self::Output {
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
pub fn get_x_unique_random_locations(count: u32, max_x: u32, max_y: u32, safe_radius: u32, taken_points: &Vec<Location>) -> Vec<Location> {
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

pub fn get_location(layout: Layout, width: u32, height: u32, entity_count: u32, safe_radius: u32) -> Vec<Location> {

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
    }

}