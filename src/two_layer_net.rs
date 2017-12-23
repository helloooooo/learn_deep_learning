extern crate nalgebra;


use std::rc::Rc;
use std::cell::RefCell;
use nalgebra::core::*;
use nural;
use lossfunc;
use gradient;


#[derive(Clone)]
pub struct grad {
    w1: DMatrix<f64>,
    b1: DMatrix<f64>,
    w2: DMatrix<f64>,
    b2: DMatrix<f64>,
}
pub struct Two_layer_network {
    w1: Rc<RefCell<DMatrix<f64>>>,
    b1: Rc<RefCell<DMatrix<f64>>>,
    w2: Rc<RefCell<DMatrix<f64>>>,
    b2: Rc<RefCell<DMatrix<f64>>>,
}
impl Two_layer_network {
    pub fn predict(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let a1 = nural::dot(&x, &self.w1.borrow(), &self.b1.borrow());
        let z1 = nural::sigmoid(&a1);
        let a2 = nural::dot(&z1, &self.w2.borrow(), &self.b2.borrow());
        let y = nural::softmax(&a2);
        y
    }
    pub fn loss(&self, x: &DMatrix<f64>, t: &DMatrix<f64>) -> f64 {
        let y = self.predict(x);
        lossfunc::cross_entropy(&y, t)
    }
    pub fn numerical_gradient(&mut self, x: &DMatrix<f64>, t: &DMatrix<f64>) -> grad {

        let grads = grad {
            w1: gradient::numerical_gradient(loss_w, &self, &self.w1, t),
            b1: gradient::numerical_gradient(loss_w, &self, &self.b1, t),
            w2: gradient::numerical_gradient(loss_w, &self, &self.w2, t),
            b2: gradient::numerical_gradient(loss_w, &self, &self.b2, t),
        };
        grads
    }
}

pub fn loss_w(x: &DMatrix<f64>, t: &DMatrix<f64>, two: &Two_layer_network) -> f64 {
    two.loss(x, t)
}
