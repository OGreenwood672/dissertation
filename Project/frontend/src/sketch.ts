// import p5 from "p5";
import { setupWebSocket } from "./websocket";
import { AgentConfig, loadConfig, SimulationConfig, StationConfig } from "./config";
import { AgentState, Message, StationState } from "./types";

const CONFIG_PATH = "http://localhost:5173/simulation.yaml"

export const sketch = (p: p5) => {

    let config: SimulationConfig;
    let configText: string[];
    let ws: WebSocket;

    let agents: StationState[] = [];
    let stations: StationState[] = [];


    p.preload = () => {
        console.log("PRELOAD");
        p.loadStrings(CONFIG_PATH, (result: string[]) => {
            console.log("SUCCESS: Config file loaded!");
            configText = result;
        }, () => {
            console.error("ERROR: Failed to load config file. Check path and network tab.");
        });
    };

    p.setup = () => {
        console.log("SETUP")

        config = loadConfig(configText.join("\n"));

        stations = [];
        agents = [];

        const ws_url = [config.websocket_url, config.websocket_path].join("/");
        ws = setupWebSocket(ws_url, (data: Message) => {
            agents = data.agents;
            stations = data.stations
        });

        p.createCanvas(config.arena_width, config.arena_height);
        p.background(0);
    };

    p.draw = () => {
        p.background(0);
        p.fill(255);

        stations.forEach((station) => {
            p.rect(station.location.x, station.location.y, config.station_size, config.station_size);
        });

        agents.forEach((agent) => {
            p.rect(agent.location.x, agent.location.y, config.agent_size, config.agent_size);
        });
    
    };
};