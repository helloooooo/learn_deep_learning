extern crate nalgebra;

use nalgebra::core::DMatrix;
use lossfunc;
use std::cell::RefCell;
#[derive(Debug, Clone)]
pub struct Neural {
    weight: DMatrix<f64>,
}
impl Neural {
    pub fn predict(self, x: &DMatrix<f64>) -> DMatrix<f64> {
        x * self.weight
    }
    pub fn loss(self, x: &DMatrix<f64>, t: &DMatrix<f64>) -> f64 {
        let z = self.predict(&x);
        let y = softmax(&z);
        println!("{:?}", y.clone());
        let loss = lossfunc::cross_entropy(&y, &t);
        loss
    }
}
pub fn dot(x: &DMatrix<f64>, w: &DMatrix<f64>, b: &DMatrix<f64>) -> DMatrix<f64> {
    x * w + b
}

pub fn sigmoid(x: &DMatrix<f64>) -> DMatrix<f64> {
    x.map(|i| 1.0 / (1.0 + (-i).exp()))
}

pub fn softmax(a: &DMatrix<f64>) -> DMatrix<f64> {
    let c = a.iter().fold(0.0 / 0.0, |acc, i| i.max(acc));
    let exp_a = a.map(|x| f64::exp(x - c));
    let sum_exp_a: f64 = exp_a.iter().sum();
    let y = exp_a.map(|_x| _x / sum_exp_a);
    y
}

#[test]
fn p111() {
    let y = DMatrix::<f64>::from_iterator(
        2,
        3,
        [
            0.47355232,
            0.85557411,
            0.9977393,
            0.03563661,
            0.84668094,
            0.69422093,
        ].iter()
            .cloned(),
    );
    let ne = Neural { weight: y };
    let _ne = ne.clone();
    let x = DMatrix::<f64>::from_iterator(1, 2, [0.6, 0.9].iter().cloned());
    let _x = x.clone();
    let ans1 = DMatrix::<f64>::from_iterator(
        1,
        3,
        [1.054148091, 0.630716529, 1.132807401].iter().cloned(),
    );
    let t = DMatrix::<f64>::from_iterator(1, 3, [0.0, 0.0, 1.0].iter().cloned());
    let ans2 = 0.92806828578640754;
    assert_eq!(ans1, ne.predict(&x));
    assert_eq!(ans2, _ne.loss(&_x, &t));
}
#[test]
fn soft() {
    let t = DMatrix::<f64>::from_iterator(1, 3, [0.3, 2.9, 4.0].iter().cloned());
    let ans = DMatrix::<f64>::from_iterator(
        1,
        3,
        [0.01821127329554753, 0.24519181293507392, 0.7365969137693786]
            .iter()
            .cloned(),
    );
    assert_eq!(ans, softmax(&t));
}
#[test]
fn sig() {
    let t = DMatrix::<f64>::from_iterator(1, 3, [-1.0, 1.0, 2.0].iter().cloned());
    let ans = DMatrix::<f64>::from_iterator(
        1,
        3,
        [0.2689414213699951, 0.7310585786300049, 0.8807970779778823]
            .iter()
            .cloned(),
    );
    assert_eq!(ans, sigmoid(&t));
}
