import { Message } from "./types";



const serialize_agents = (data: string) => {


    let msg: Message = JSON.parse(data) as Message;

    return msg;

}

export const setupWebSocket = (url: string, process_data: (data: any) => void): WebSocket => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
        console.log("Connected to WebSocket server");
    };

    ws.onmessage = (event) => {
        // console.log("Received:", event.data);
        process_data(serialize_agents(event.data));
    };

    ws.onclose = () => {
        console.log("Disconnected from WebSocket server");
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
    };

    return ws;
};
