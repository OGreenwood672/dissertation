// import p5 from "p5";
import { setupWebSocket } from "./websocket";
import { loadConfig, Config } from "./config";
import { Message, WorldState } from "./types";

const CONFIG_PATH = "http://localhost:5173/frontend_config.json"

export const sketch = (p: p5) => {

    let config: Config;
    let configText: string[];
    let ws: WebSocket;

    let worlds: WorldState[] = [];

    let WIDTH = p.windowWidth * 0.95;
    let HEIGHT = p.windowHeight * 0.95;

    const STATION_SIZE = 5;
    const AGENT_SIZE = 2;

    let aspect = WIDTH / HEIGHT;

    let cols: number, rows: number;
    let cell_w: number, cell_h: number;
    let square_size: number;
    let scale: number;

    p.preload = () => {
        console.log("Preloading...");

        p.loadStrings(CONFIG_PATH, (result: string[]) => {
            console.log("SUCCESS: Config file loaded!");
            configText = result;
        }, () => {
            console.error("ERROR: Failed to load config file");
        });

        console.log("Preload complete");
    };

    p.setup = () => {
        console.log("setting up...");

        config = loadConfig(configText.join("\n"));

        cols = Math.ceil(Math.sqrt(config.worlds_parallised * aspect));
        rows = Math.ceil(config.worlds_parallised / cols);

        cell_w = WIDTH / cols;
        cell_h = HEIGHT / rows;

        square_size = Math.min(cell_w, cell_h);
        scale = Math.min(square_size / config.arena_width, square_size / config.arena_height);

        WIDTH = square_size * cols;
        HEIGHT = square_size * rows;

        worlds = Array.from({ length: config.worlds_parallised }, () => ({} as WorldState));

        const ws_url = [config.websocket_url, config.websocket_path].join("/");
        ws = setupWebSocket(ws_url, (data: Message) => {
            worlds[data.world_id] = data.world_state;
        });

        p.createCanvas(WIDTH + 1, HEIGHT + 1);
        p.background(0);

        console.log("setup complete");
    };

    p.draw = () => {
        p.background(0);
        p.fill(255);

        // console.log(config.worlds_parallised, cols, rows);

        if (!(cols && rows && cell_w && cell_h && square_size && scale)) { return; }

        for (let y = 0; y < rows; y++) {
            for (let x = 0; x < cols; x++) {

                const index = y * cols + x;

                if (index < worlds.length) {
                    const world = worlds[index];

                    let cell_x = x * square_size;
                    let cell_y = y * square_size;

                    p.push();
                    p.noFill();
                    p.strokeWeight(1);
                    p.stroke(255);
                    p.rect(cell_x, cell_y, square_size, square_size);
                    p.pop();

                    if (world.agents && world.stations) {
                        world.agents.forEach((agent) => {
                            p.rect(
                                agent.location.x * scale + cell_x - AGENT_SIZE / 2,
                                agent.location.y * scale + cell_y - AGENT_SIZE / 2,
                                AGENT_SIZE * scale,
                                AGENT_SIZE * scale
                            );
                        });
                        world.stations.forEach((station) => {
                            p.rect(
                                station.location.x * scale + cell_x - STATION_SIZE / 2,
                                station.location.y * scale + cell_y - STATION_SIZE / 2,
                                STATION_SIZE * scale,
                                STATION_SIZE * scale
                            );
                        });
                    }
                }
            }
        }
    
    };
};