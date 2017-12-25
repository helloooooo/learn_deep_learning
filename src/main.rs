extern crate mnist;
extern crate nalgebra;
extern crate gnuplot;
extern crate rand;
use std::rc::Rc;
use std::cell::RefCell;

use mnist::{Mnist, MnistBuilder};
use nalgebra::core::DMatrix;

use gnuplot::{Figure, Caption, Color, AxesCommon, Fix};
use rand::Rng;

mod lossfunc;
mod gradient;
mod nural;
mod two_layer_net;


fn main() {
    let (size, rows, cols) = (60_000, 28, 28);
    let mut train_loss_list = Vec::new();
    // Deconstruct the returned Mnist struct.
    let Mnist { trn_img, trn_lbl, .. } = MnistBuilder::new().label_format_one_hot()
        //.label_format_digit()
        .training_set_length(size)
        .finalize();

    let x_vec = trn_img.iter().map(|i| *i as f64).collect::<Vec<f64>>();
    let t_vec = trn_lbl.iter().map(|i| *i as f64).collect::<Vec<f64>>();
    let x = DMatrix::<f64>::from_row_slice(60000, 784, &x_vec);
    let t = DMatrix::<f64>::from_row_slice(60000, 10, &t_vec);
    let mut Two_layer_network = two_layer_net::Two_layer_network {
        w1: Rc::new(RefCell::new(DMatrix::<f64>::new_random(784, 100))),
        b1: Rc::new(RefCell::new(DMatrix::<f64>::zeros(100, 100))),
        w2: Rc::new(RefCell::new(DMatrix::<f64>::new_random(100, 10))),
        b2: Rc::new(RefCell::new(DMatrix::<f64>::zeros(100, 10))),
    };
    let iters_num = 10000;
    let batch_size = 100;
    let learning_late = 0.1;
    let mut vec = Vec::new();
    for n in 0..iters_num {
        println!("{}",n);
        vec.push(n);
        let rand_number = rand::thread_rng().gen_range(1, 59900) as usize;
        let x_batch = &x.rows(rand_number, 100).clone_owned();
        let t_batch = &t.rows(rand_number, 100).clone_owned();
        let grad = Two_layer_network.gradient(x_batch, t_batch);
        *Two_layer_network.w1.borrow_mut() = two_layer_net::learn(&learning_late, &grad.w1);
        *Two_layer_network.b1.borrow_mut() = two_layer_net::learn(&learning_late, &grad.b1);
        *Two_layer_network.w2.borrow_mut() = two_layer_net::learn(&learning_late, &grad.w2);
        *Two_layer_network.b2.borrow_mut() = two_layer_net::learn(&learning_late, &grad.b2);
        let sub = DMatrix::<f64>::zeros(100, 100);
        let loss = Two_layer_network.loss(&sub, &x_batch, &t_batch, "nothing");
        train_loss_list.push(loss);
    }
    println!("{:?}", train_loss_list);
   
    let mut fg = Figure::new();
     fg.axes2d()
         .lines(&vec, &train_loss_list, &[Caption("test"), Color("red")])
         .set_y_range(Fix(-1000.0), Fix(1000.0));
     fg.show();
}
