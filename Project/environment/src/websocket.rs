use tokio::sync::{broadcast};
use warp::{filters::ws::{Message, Ws}, Filter};
use futures_util::{SinkExt, StreamExt};


pub fn websocket(
    tx: broadcast::Sender<String>,
) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path("ws")
        .and(warp::ws())
        .map(move |ws: Ws| {
            let mut rx = tx.subscribe();
            ws.on_upgrade(move |socket: warp::filters::ws::WebSocket| async move {
                // Relay incoming channel messages to the websocket client
                let (mut ws_tx, _) = socket.split();
                while let Ok(msg) = rx.recv().await {
                    if ws_tx.send(Message::text(msg)).await.is_err() {
                        eprintln!("WebSocket send error or client disconnected");
                        break;
                    }
                }
            })
        })
}