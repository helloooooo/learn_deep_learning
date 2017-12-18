extern crate nalgebra;

use nalgebra::core::DMatrix;
use lossfunc;
#[derive(Debug)]
pub struct Neural {
    weight: DMatrix<f64>,
}
impl Neural {
    pub fn predict(self, x: DMatrix<f64>) -> DMatrix<f64> {
        x * self.weight
    }
    pub fn loss(self, x: DMatrix<f64>, t: DMatrix<f64>) -> f64 {
        let z = self.predict(x);
        let y = softmax(z);
        let loss = lossfunc::cross_entropy(y, t);
        loss
    }
}
pub fn softmax(a: DMatrix<f64>) -> DMatrix<f64> {
    let c = a.iter().fold(0.0 / 0.0, |acc, i| i.max(acc));
    let exp_a = a.map(|x| f64::exp(x - c));
    let sum_exp_a: f64 = exp_a.iter().sum();
    let y = exp_a.map(|_x| _x / sum_exp_a);
    y
}

#[test]
fn p111() {
    let t = DMatrix::<f64>::from_iterator(
        3,
        2,
        [
            0.47355232,
            0.9977393,
            0.84668094,
            0.85557411,
            0.03563611,
            0.69422093,
        ].iter()
            .cloned(),
    );
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
    let x = DMatrix::<f64>::from_iterator(1, 2, [0.6, 0.9].iter().cloned());
    let ans1 = DMatrix::<f64>::from_iterator(
        1,
        3,
        [1.054148091, 0.630716529, 1.132807401].iter().cloned(),
    );
    assert_eq!(ans1, ne.predict(x));
}
