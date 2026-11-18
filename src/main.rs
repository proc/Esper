//! Esper - a UCI chess engine.

mod bitboard;
mod movegen;
mod nnue;
mod position;
mod search;
mod tb;
mod tune;
mod tt;
mod uci;

fn main() {
    bitboard::init();
    let net = match uci::load_default_network() {
        Ok(n) => n,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let mut engine = uci::Engine::new(net);
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `-t N` / `--threads N` sets the number of search threads.
    while let Some(i) = args.iter().position(|a| a == "-t" || a == "--threads") {
        match args.get(i + 1).and_then(|v| v.parse::<usize>().ok()) {
            Some(n) if n >= 1 => engine.set_default_threads(n),
            _ => {
                eprintln!("error: {} needs a thread count of at least 1", args[i]);
                std::process::exit(2);
            }
        }
        args.drain(i..i + 2);
    }
    if args.is_empty() {
        engine.run();
    } else {
        // Commands given on the command line, e.g. `esper bench`.
        engine.command(&args.join(" "));
        engine.command("quit");
    }
}
