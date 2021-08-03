#![cfg_attr(all(feature = "nightly", test), feature(test))]
#[macro_use]
extern crate itertools;
extern crate ndarray;
extern crate num_traits;

#[cfg_attr(test, macro_use)]
extern crate float_cmp;

#[cfg(test)]
extern crate rand;
#[macro_use]
extern crate log;

extern crate all_lap_rust;
#[cfg(all(feature = "nightly", test))]
extern crate test;
pub mod allenum;
pub mod kbest;
pub mod lapjv;
