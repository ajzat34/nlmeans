use std::path::PathBuf;
use image::{RgbImage, buffer::ConvertBuffer};
use clap::Parser;

mod denoise;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    // input file
    #[arg(short, long, value_name = "INPUT FILE")]
    input: PathBuf,

    // output file
    #[arg(short, long, value_name = "OUTPUT FILE")]
    output: PathBuf,

    // sample radius
    #[arg(short, long, default_value_t = 2)]
    patch_size: u32,

    // search radius
    #[arg(short, long, default_value_t = 8)]
    search_size: u32,

    // filtering param
    #[arg(short, long, default_value_t = 20.0)]
    filtering_parameter: f32,
}

fn main() {
    let args = Args::parse();

    let input = args.input;
    let output = args.output;

    let sample_radius = args.patch_size;
    let search_radius = args.search_size;
    let filtering = args.filtering_parameter;

    let img = image::open(input).unwrap().to_rgb32f();
    let result = denoise::nlmeans(&img, sample_radius, search_radius, filtering);
    let out: RgbImage = result.convert();
    out.save(output).unwrap();
}