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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast;
    use warp::test::ws;

    #[tokio::test]
    async fn test_websocket_relays_broadcast_messages() {
        let (tx, _rx) = broadcast::channel(16);
        let filter = websocket(tx.clone());
        let mut client = ws()
            .path("/ws")
            .handshake(filter)
            .await
            .expect("handshake");

        let test_msg = "hello from rust".to_string();
        tx.send(test_msg.clone()).unwrap();

        let msg = tokio::time::timeout(std::time::Duration::from_millis(100), client.recv())
            .await
            .expect("Timeout waiting for message")
            .expect("Client received None");

        assert!(msg.is_text());
        assert_eq!(msg.to_str().unwrap(), test_msg);
    }

    #[tokio::test]
    async fn test_websocket_handles_multiple_messages() {
        let (tx, _rx) = broadcast::channel(16);
        let filter = websocket(tx.clone());

        let mut client = ws()
            .path("/ws")
            .handshake(filter)
            .await
            .expect("handshake");

        tx.send("MSG".into()).unwrap();
        tx.send("SECOND MSG".into()).unwrap();

        let binding = client.recv().await.unwrap();
        let m1 = binding.to_str().unwrap();
        let binding = client.recv().await.unwrap();
        let m2 = binding.to_str().unwrap();

        assert_eq!(m1, "MSG");
        assert_eq!(m2, "SECOND MSG");
    }
}