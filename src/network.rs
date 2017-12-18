extern crate nalgebra;
use nalgebra::core::{DMatrix};

pub struct Network{
    b: DMatrix<f64>,
    data: DMatrix<f64>,
    weight: DMatrix<f64>,
}

impl Network{
    pub fn dot(&self) -> DMatrix<f64>{
        self.data * self.weight + self.b
    }
    pub fn sigmoid(&self) -> DMatrix<f64>{
        self.dot().map(|i| {1.0 /(1.0 + (-i).exp())})
    }
    pub fn softmax(&self) -> DMatrix<f64> {
        let _dot = self.dot();
        let _max = _dot.iter().fold(0.0 / 0.0, |acc, i| i.max(acc) ); // NaNでないものを返す.
        let _sum = _dot.iter().fold(0., |acc, i| (i - _max).exp() + acc);
        _dot.map(|i| { (i - _max).exp() / _sum })
    }
}