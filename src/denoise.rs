use image::{GenericImage, GenericImageView, Pixel, Rgb32FImage, ImageBuffer, Primitive};
use statrs::distribution::{Normal, Continuous};

pub fn nlmeans(
    inp_image: &Rgb32FImage,// input image
    sample_radius: u32,     // half size of comparison area (minus 1)
    search_radius: u32,     // half size of research window (minus 1)
    filter_param: f32,      // filtering parameter
) -> Rgb32FImage
{   
    // input image dimensions
    let (inp_width, inp_height) = inp_image.dimensions();

    // full size of windows
    let sample_size = 2 * sample_radius + 1;
    let search_size = 2 * search_radius + 1;
    
    let i_search_radius = search_radius as i32;
    let i_search_size = search_size as i32;

    let fp = (filter_param/255.0) * (sample_size as f32);
    let fp2 = fp*fp;
    let fp2inv = 1.0/fp2;

    // distance from pixel to furthest compared pixel one one axis
    let offset_size = sample_radius+search_radius;

    // pad original image so search window doesnt go out of bounds
    let img = pad_image(inp_image, offset_size);
    let mut out = Rgb32FImage::new(inp_width, inp_height);

    let usearch = search_size as usize;
    let mut weights_buff = vec![vec![0.0; usearch]; usearch];

    // create a gaussian distrabution LUT
    let mut g_lut = vec![vec![0.0; usearch]; usearch];
    let n = Normal::new(0.0, search_radius as f64).unwrap();
    for iy in 0..i_search_size as i32 {
        for ix in 0..i_search_size {
            let fx = (ix-i_search_radius) as f64;
            let fy = (iy-i_search_radius) as f64;
            let dist = (fx*fx + fy*fy).sqrt();
            let v = n.pdf(dist) as f32;
            g_lut[iy as usize][ix as usize] = v;
        }
    }

    // per pixel denoising
    for opy in 0..inp_height {
        if opy % (inp_height/100) == 0 { println!("{}%", (opy*100)/inp_height); }
        
        for opx in 0..inp_width {

            // account for padding
            let px = opx + offset_size;
            let py = opy + offset_size;

            // get similarity values for neighboring pixels
            let mut q_acc = 0.0;
            for iy in 0..search_size {
                for ix in 0..search_size {
                    // center the search area on the pixel
                    let cx = px-search_radius+ix;
                    let cy = py-search_radius+iy;

                    let uix = ix as usize;
                    let uiy = iy as usize;
                    // check the gaussian table
                    let gauss = g_lut[uiy][uix];

                    // how similar this area is
                    let q = gauss*compare_samples(&img, px, py, cx, cy, sample_size, sample_radius, fp2inv);
                    weights_buff[uiy][uix] = q;
                    q_acc += q;
                }
            }

            let out_pixel = out.get_pixel_mut(opx, opy).channels_mut();

            // recalculate value of this pixel as weighted sum of neighboring pixels
            for iy in 0..search_size {
                for ix in 0..search_size {
                    for i in 0..=2 {
                        let pixel = img.get_pixel(px-search_radius+ix, py-search_radius+iy).channels();
                        let q = weights_buff[iy as usize][ix as usize];
                        out_pixel[i] += pixel[i] * q;
                    }
                }
            }

            for i in 0..=2 {
                out_pixel[i] = f32::clamp(out_pixel[i] / q_acc, 0.0, 1.0);
            }
        }
    }

    out
}

fn compare_samples(
    img: &Rgb32FImage,
    ax: u32,        // center of window a
    ay: u32,
    bx: u32,        // center of window b
    by: u32,
    size: u32,      // width/height of sample
    radius: u32,    // half the width of the sample size (minus 1)
    fp2inv: f32,    // inverse of filtering parameter squared
) -> f32
{
    if (ax == bx) && (ay == by) {return 1.0;}

    let a = img.view(ax-radius, ay-radius, size, size);
    let b = img.view(bx-radius, by-radius, size, size);

    let mut sum = 0.0;
    for iy in 0..size {
        for ix in 0..size {
            let pa = a.get_pixel(ix, iy);
            let pb = b.get_pixel(ix, iy);
            for i in 0..=2 {
                let diff = pa[i] - pb[i];
                sum += diff*diff;
            }
        }
    }

    std::f32::consts::E.powf(-sum*fp2inv)
}

pub fn pad_image<I, P, S>(img: &I, padding: u32) -> ImageBuffer<P, Vec<S>>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel<Subpixel = S> + 'static,
    S: Primitive + 'static,
{
    let padding2 = padding*2;
    let mut result = ImageBuffer::new(img.width() + padding2, img.height() + padding2);
    let _ = result.copy_from(img, padding, padding);
    result
}