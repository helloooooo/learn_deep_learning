extern crate mnist;
extern crate nalgebra;
extern crate gnuplot;
mod lossfunc;
mod gradient;
mod nural;
mod two_layer_net;
use mnist::{Mnist, MnistBuilder};
use nalgebra::core::DMatrix;
use gnuplot::{Figure, Caption, Color, AxesCommon, Fix};




fn main() {
    let (size, rows, cols) = (50_000, 28, 28);

    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(size)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();
    let x1 = DMatrix::<f64>::from_iterator(1, trn_img.len(), trn_img.iter().map(|i| *i as f64));
    println!("{}", x1);
}
