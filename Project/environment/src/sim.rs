use pyo3::{pyfunction, PyResult};
use tokio::runtime::Runtime;
use tokio::signal;
use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use crate::config::{load_config};
use crate::world::World;
use crate::websocket::websocket;


#[pyfunction]
pub fn run_sim_blocking(config_path: String) -> PyResult<()> {
    let rt = Runtime::new().unwrap();
    rt.block_on(run_sim(config_path));
    Ok(())
}

async fn run_sim(config_path: String) {

    let cancel_token = CancellationToken::new();
    let cancel_token_ping = cancel_token.clone();
    let cancel_token_ws = cancel_token.clone();

    println!("starting sim");

    let config = match load_config(config_path) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("[ERROR] unable to load config: {}", err);
            return;
        }
    };

    World::new(config);

    let (bcast_tx, _) = broadcast::channel::<String>(32);
    // Clone for broadcast task
    let bcast_tx2 = bcast_tx.clone();

    // Task for broadcasting pings
    let ping_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancel_token_ping.cancelled() => {
                    println!("Ping task received cancellation!");
                    break;
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {
                    match bcast_tx2.send("ping".to_string()) {
                        Ok(_) => println!("broadcasted ping"),
                        Err(e) => println!("Error broadcasting: {}", e)
                    }
                }
            }
        }
    });

    // WS server task
    let ws_filter = websocket(bcast_tx.clone());
    let ws_task = tokio::spawn(async move {
        warp::serve(ws_filter)
        .bind(([127, 0, 0, 1], 3000)).await
        .graceful(async move {
            signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
            cancel_token_ws.cancel();
        })
        .run().await;
    });

    // Wait for tasks (or use tokio::try_join! to error-propagate)
    let (ping_res, ws_res) = tokio::join!(ping_task, ws_task);
    println!("Ping task result: {:?}", ping_res);
    println!("WS task result: {:?}", ws_res);

}