
export interface Location {
    x: number;
    y: number;
}

export interface AgentState {
    id: number;
    location: Location;
}

export interface StationState {
    id: number;
    location: Location;
}

export interface WorldState {
    agents: AgentState[];
    stations: StationState[];
}


export interface Message {
    world_id: number;
    world_state: WorldState;
}