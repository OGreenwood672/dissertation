import p5 from "p5";
import { sketch } from "./src/sketch";
import { setupWebSocket } from "./src/websocket";

const URL = "ws://127.0.0.1:3000/ws";

const ws = setupWebSocket(URL);

new p5(sketch);
