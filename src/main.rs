use crate::problem_500::license_key_formatting;

mod common;
mod problem_100;
mod problem_200;
mod problem_300;
mod problem_400;
mod problem_500;

fn main() {
    println!(
        "{:?}",
        license_key_formatting(String::from("2-4A0r7-4k"), 3)
    );
}
