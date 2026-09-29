//! Round-trip latency of waking a thread that is blocked on another, idle core
//! (`thread::park`/`unpark`, i.e. `sys_futex_wait`/`sys_futex_wake`).
//!
//! The main thread runs on core 0 and the spawned worker on core 1 (the round-robin
//! core selection of the kernel starts at 1). Both threads block while waiting for
//! their turn. Each thread busy-waits for the hand-over delay before handing over, so
//! the partner core has left the idle backoff and sleeps in `wfi`. Every hand-over
//! therefore wakes an idle core with exactly one IPI.
//!
//! Usage: `wakeup_benchmark [--rounds N] [--handover-delay-us N]`. With a hand-over
//! delay shorter than the idle backoff in `PerCoreScheduler::run`, the partner core
//! still spins and is woken up without an IPI. Counters collected outside the guest
//! cover a whole run including boot and warm-up, so compare two runs with different
//! numbers of rounds to attribute them to the measured rounds.
//!
//! With the `event-log` feature, the kernel logs its IPI events of the whole run.
//! `window_start_ticks` and `window_end_ticks` bound the measured rounds in ticks of
//! the `time` counter, so the events can be attributed to them. Both are 0 on other
//! architectures than riscv64.
//!
//! Requires at least 2 cores. With the `idle-poll` feature, idle cores poll instead
//! of sleeping and no IPIs are sent. Comparing both builds yields the share of the
//! latency caused by the IPI.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::{env, hint, thread};

#[cfg(target_os = "hermit")]
use hermit as _;
#[cfg(target_arch = "riscv64")]
use riscv::register::time;

const WARMUP_ROUNDS: u64 = 50;
const ROUNDS: u64 = 5000;

/// Longer than the idle backoff in `PerCoreScheduler::run`, so the partner core
/// is in `wfi` and has to be woken by an IPI.
const HANDOVER_DELAY_US: u64 = 100;

/// Odd values hand the turn to the worker, even values back to the main thread.
static TURN: AtomicU64 = AtomicU64::new(0);

fn wait_for_turn(turn: u64) {
	while TURN.load(Ordering::Acquire) != turn {
		thread::park();
	}
}

/// Reads the `time` counter, which the kernel uses for the timestamps of its event log.
#[cfg(target_arch = "riscv64")]
fn time_ticks() -> u64 {
	time::read64()
}

#[cfg(not(target_arch = "riscv64"))]
fn time_ticks() -> u64 {
	0
}

/// Spins instead of sleeping, so no timer interrupts are involved.
fn busy_wait(delay: Duration) {
	let start = Instant::now();
	while start.elapsed() < delay {
		hint::spin_loop();
	}
}

fn main() {
	let mut rounds = ROUNDS;
	let mut handover_delay_us = HANDOVER_DELAY_US;

	let mut args = env::args().skip(1);
	while let Some(arg) = args.next() {
		match arg.as_str() {
			"--rounds" => {
				rounds = args
					.next()
					.expect("--rounds requires a value.")
					.parse()
					.expect("The number of rounds is invalid.");
			}
			"--handover-delay-us" => {
				handover_delay_us = args
					.next()
					.expect("--handover-delay-us requires a value.")
					.parse()
					.expect("The hand-over delay is invalid.");
			}
			_ => panic!("The argument {arg} is unknown."),
		}
	}
	assert!(rounds > 0, "This benchmark requires at least 1 round.");
	let handover_delay = Duration::from_micros(handover_delay_us);

	assert!(
		std::thread::available_parallelism().unwrap().get() >= 2,
		"This benchmark requires at least 2 cores."
	);

	let total_rounds = WARMUP_ROUNDS + rounds;

	let main_thread = thread::current();
	let worker = thread::spawn(move || {
		for i in 0..total_rounds {
			wait_for_turn(2 * i + 1);
			busy_wait(handover_delay);
			TURN.store(2 * i + 2, Ordering::Release);
			main_thread.unpark();
		}
	});

	let mut window_start = 0;
	let mut start = Instant::now();
	for i in 0..total_rounds {
		if i == WARMUP_ROUNDS {
			window_start = time_ticks();
			start = Instant::now();
		}
		busy_wait(handover_delay);
		TURN.store(2 * i + 1, Ordering::Release);
		worker.thread().unpark();
		wait_for_turn(2 * i + 2);
	}
	let elapsed = start.elapsed();
	let window_end = time_ticks();

	worker.join().unwrap();

	// Each round contains two hand-over delays, which are not part of the round trip.
	println!(
		"wakeup_benchmark rounds={rounds} handover_delay_us={handover_delay_us} total_us={} mean_round_trip_ns={} window_start_ticks={window_start} window_end_ticks={window_end}",
		elapsed.as_micros(),
		elapsed.as_nanos() / u128::from(rounds) - 2 * handover_delay.as_nanos()
	);
}
