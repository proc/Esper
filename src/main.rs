//! Esper - a UCI chess engine.

mod bitboard;
mod book;
mod cuckoo;
mod datagen;
mod exp;
mod mem;
mod movegen;
mod nnue;
mod pgn;
mod polyglot_keys;
mod position;
mod search;
mod simd;
mod small;
mod tb;
mod tune;
mod tt;
mod uci;

fn main() {
    // A build made for a newer CPU than this one would otherwise die on an
    // illegal instruction with no explanation.
    if let Some(feature) = simd::missing_baseline() {
        eprintln!("error: this build of Esper needs a CPU with {feature}, which this one lacks");
        std::process::exit(1);
    }
    bitboard::init();
    cuckoo::init();
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
