//! Overshoot of timed sleeps over a sweep of intervals (`thread::park_timeout`,
//! i.e. `sys_futex_wait` with a relative timeout).
//!
//! `thread::sleep` is not used, because `sys_usleep` busy-waits for durations below
//! 10 ms and arms no timer. A futex wait with a timeout always blocks the task and
//! arms the one-shot timer. The core then runs the idle task and sleeps in `wfi`,
//! until the timer interrupt wakes up the task. Nobody unparks the main thread, so
//! every round ends with a timeout.
//!
//! The overshoot is the time spent in `park_timeout` beyond the requested interval.
//! It is reported with two lines per interval: summary statistics and a histogram.
//! The histogram uses log2 buckets in microseconds. Each key is the lower bound of
//! its bucket, `lt0` counts rounds that returned early.
//!
//! Usage: `sleep_benchmark [--rounds N] [INTERVAL_US ...]`. Without intervals, the
//! whole sweep runs. Counters collected outside the guest cover a whole run, so
//! pass a single interval to attribute them to one interval.
//!
//! Limits of the measurement:
//! - The kernel keeps deadlines in whole microseconds and `Instant` has a resolution
//!   of 1 us. A single sample is therefore only accurate to about 1 us.
//! - The kernel wakes up a task once the current time is greater than its deadline,
//!   so the smallest overshoot is 1 us. If the timer interrupt is handled within
//!   the microsecond of the deadline, the kernel re-arms the timer with the expired
//!   deadline and takes further timer interrupts until the time advances.
//! - Before sleeping in `wfi`, the idle task spins through the idle backoff in
//!   `PerCoreScheduler::run`. At 100 us it takes up a relevant share of the interval.
//!
//! With the `idle-poll` feature, the idle core polls instead of sleeping in `wfi`.
//! Comparing both builds yields the share of the overshoot caused by waking up the
//! core.
//!
//! With the `event-log` feature, the kernel logs its timer events of the whole run.
//! `window_start_ticks` and `window_end_ticks` bound the measured rounds of an
//! interval in ticks of the `time` counter, so the events can be attributed to them.
//! Both are 0 on other architectures than riscv64.

use std::fmt::Write;
use std::time::{Duration, Instant};
use std::{env, thread};

#[cfg(target_os = "hermit")]
use hermit as _;
#[cfg(target_arch = "riscv64")]
use riscv::register::time;

const WARMUP_ROUNDS: usize = 50;

const ROUNDS: usize = 2000;

const INTERVALS_US: [u64; 3] = [100, 1_000, 10_000];

/// Reads the `time` counter, which the kernel uses for the timestamps of its event log.
#[cfg(target_arch = "riscv64")]
fn time_ticks() -> u64 {
	time::read64()
}

#[cfg(not(target_arch = "riscv64"))]
fn time_ticks() -> u64 {
	0
}

/// Returns the overshoot of every round in nanoseconds and the start and end of the
/// measured rounds in ticks of the `time` counter. The overshoot is negative if
/// `park_timeout` returned early.
fn measure(interval: Duration, rounds: usize) -> (Vec<i64>, (u64, u64)) {
	let interval_ns = i64::try_from(interval.as_nanos()).unwrap();

	let mut window_start = 0;
	let mut samples = Vec::with_capacity(rounds);
	for i in 0..WARMUP_ROUNDS + rounds {
		if i == WARMUP_ROUNDS {
			window_start = time_ticks();
		}

		let start = Instant::now();
		thread::park_timeout(interval);
		let elapsed = start.elapsed();

		if i >= WARMUP_ROUNDS {
			samples.push(i64::try_from(elapsed.as_nanos()).unwrap() - interval_ns);
		}
	}
	(samples, (window_start, time_ticks()))
}

/// Nearest-rank percentile of sorted samples.
fn percentile(sorted: &[i64], percent: usize) -> i64 {
	sorted[(sorted.len() * percent).div_ceil(100) - 1]
}

fn report(interval: Duration, mut samples: Vec<i64>, window: (u64, u64)) {
	samples.sort_unstable();

	let rounds = samples.len();
	let early = samples.iter().filter(|&&sample| sample < 0).count();
	let mean = samples.iter().sum::<i64>() / i64::try_from(rounds).unwrap();

	println!(
		"sleep_benchmark interval_us={} rounds={rounds} early={early} min_overshoot_ns={} mean_overshoot_ns={mean} p50_overshoot_ns={} p90_overshoot_ns={} p99_overshoot_ns={} max_overshoot_ns={} window_start_ticks={} window_end_ticks={}",
		interval.as_micros(),
		samples[0],
		percentile(&samples, 50),
		percentile(&samples, 90),
		percentile(&samples, 99),
		samples[rounds - 1],
		window.0,
		window.1
	);

	// Bucket 0 covers [0, 1) us, bucket n covers [2^(n-1), 2^n) us.
	let mut buckets: Vec<usize> = Vec::new();
	for &sample in &samples[early..] {
		let overshoot_us = sample.unsigned_abs() / 1000;
		let bucket = match overshoot_us {
			0 => 0,
			us => us.ilog2() as usize + 1,
		};
		if buckets.len() <= bucket {
			buckets.resize(bucket + 1, 0);
		}
		buckets[bucket] += 1;
	}

	// Built as a single line, so it is written to the console at once.
	let mut line = format!(
		"sleep_benchmark_hist interval_us={} unit=us lt0={early}",
		interval.as_micros()
	);
	for (bucket, count) in buckets.iter().enumerate() {
		let lower_bound_us = match bucket {
			0 => 0,
			n => 1u64 << (n - 1),
		};
		write!(line, " {lower_bound_us}={count}").unwrap();
	}
	println!("{line}");
}

fn main() {
	let mut rounds = ROUNDS;
	let mut intervals_us = Vec::new();

	let mut args = env::args().skip(1);
	while let Some(arg) = args.next() {
		if arg == "--rounds" {
			rounds = args
				.next()
				.expect("--rounds requires a value.")
				.parse()
				.expect("The number of rounds is invalid.");
		} else {
			intervals_us.push(arg.parse().expect("The interval is invalid."));
		}
	}
	assert!(rounds > 0, "This benchmark requires at least 1 round.");
	if intervals_us.is_empty() {
		intervals_us.extend(INTERVALS_US);
	}

	for interval_us in intervals_us {
		let interval = Duration::from_micros(interval_us);
		let (samples, window) = measure(interval, rounds);
		report(interval, samples, window);
	}
}
