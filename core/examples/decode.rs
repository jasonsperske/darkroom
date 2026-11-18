//! Decode a file from the command line and print what we found: `cargo run --example decode -- photo.nef`
fn main() {
    let path = std::env::args().nth(1).expect("usage: decode <file>");
    let bytes = std::fs::read(&path).unwrap();
    let t = std::time::Instant::now();
    match darkroom_core::decode::decode(&bytes, &path) {
        Ok(d) => println!("{} × {} · {} · orientation {} · {:?} · {:?}", d.width, d.height, d.description, d.orientation, d.transfer, t.elapsed()),
        Err(e) => println!("error: {e}"),
    }
}
