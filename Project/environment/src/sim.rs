use pyo3::{pyfunction, PyResult};
use tokio::runtime::Runtime;
use tokio::signal;
use serde_json;
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
    let cancellation_rx = cancel_token.clone();

    println!("starting sim");

    let config = match load_config(config_path) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("[ERROR] unable to load config: {}", err);
            return;
        }
    };

    let world = World::new(config);

    let (bcast_tx, _) = broadcast::channel::<String>(32);
    let bcast_tx2 = bcast_tx.clone();

    let update_task = tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = cancellation_rx.cancelled() => {
                    println!("Ping task received cancellation!");
                    break;
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(1000)) => {
                    let world_state = world.get_state();
                    let json_state = serde_json::to_string(&world_state)
                        .expect("Failed to serialize agent states to JSON");

                    match bcast_tx2.send(json_state) {
                        Ok(_) => { println!("broadcasted agent states") },
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
            cancel_token.cancel(); // The original token is moved here
        })
        .run().await;
    });

    // Wait for tasks (or use tokio::try_join! to error-propagate)
    let (ping_res, ws_res) = tokio::join!(update_task, ws_task);
    println!("Ping task result: {:?}", ping_res);
    println!("WS task result: {:?}", ws_res);

}