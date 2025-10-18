import { parse } from 'yaml';

export interface AgentConfig {
  id: number;
  inputs: string[];
  output: string;
}

export interface StationConfig {
  id: number;
  type: "pickup" | "dropoff";
  resource: string;
}

export interface SimulationConfig {
  arena_width: number;
  arena_height: number;
  headless: boolean;
  websocket_url: string;
  websocket_path: string;
  agent_size: number;
  initial_agent_layout: string;
  agents: AgentConfig[];
  station_size: number;
  station_layout: string;
  stations: StationConfig[];
}

export const loadConfig = (configText: string) => {
    return parse(configText) as SimulationConfig;
};
