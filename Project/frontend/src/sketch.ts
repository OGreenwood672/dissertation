// import p5 from "p5";
import { setupWebSocket } from "./websocket";
import { AgentConfig, loadConfig, SimulationConfig, StationConfig } from "./config";
import { AgentState, Message, StationState, WorldState } from "./types";

const CONFIG_PATH = "http://localhost:5173/simulation.yaml"

const WIDTH = 600;
const HEIGHT = 600;

export const sketch = (p: p5) => {

    let config: SimulationConfig;
    let configText: string[];
    let ws: WebSocket;

    let worlds: WorldState[] = [];

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

        worlds = Array.from({ length: config.worlds_parellised }, () => ({} as WorldState));

        const ws_url = [config.websocket_url, config.websocket_path].join("/");
        ws = setupWebSocket(ws_url, (data: Message) => {
            worlds[data.world_id] = data.world_state;
        });

        p.createCanvas(WIDTH, HEIGHT);
        p.background(0);
    };

    p.draw = () => {
        p.background(0);
        p.fill(255);

        let squares_per_side = Math.ceil(Math.sqrt(worlds.length));
        let square_size = Math.min(WIDTH / squares_per_side, HEIGHT / squares_per_side);
        let scale = Math.min(square_size / config.arena_width, square_size / config.arena_height);

        for (let x = 0; x < squares_per_side; x++) {
            for (let y = 0; y < squares_per_side; y++) {

                const index = x * squares_per_side + y;

                if (index < worlds.length) {

                    const world = worlds[index];
                    let x_offset = x * (WIDTH / squares_per_side);
                    let y_offset = y * (HEIGHT / squares_per_side);

                    if (world.agents && world.stations) {
                        world.agents.forEach((agent) => {
                            p.rect(
                                agent.location.x * scale + x_offset,
                                agent.location.y * scale + y_offset,
                                config.agent_size * scale,
                                config.agent_size * scale
                            );
                        });
                        world.stations.forEach((station) => {
                            p.rect(
                                station.location.x * scale + x_offset,
                                station.location.y * scale + y_offset,
                                config.station_size * scale,
                                config.station_size * scale
                            );
                        });
                    }
                }
            }
        }
    
    };
};