
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

export interface Message {
    agents: AgentState[];
    stations: StationState[];
}