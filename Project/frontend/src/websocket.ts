

export const setupWebSocket = (url: string): WebSocket => {
    const ws = new WebSocket(url);

    ws.onopen = () => {
        console.log("Connected to WebSocket server");
    };

    ws.onmessage = (event) => {
        console.log("Received:", event.data);
    };

    ws.onclose = () => {
        console.log("Disconnected from WebSocket server");
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
    };

    return ws;
};
