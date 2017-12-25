extern crate nalgebra;


use std::rc::Rc;
use std::cell::RefCell;
use nalgebra::core::*;
use nural;
use lossfunc;
use gradient;


#[derive(Clone)]
pub struct grad {
    pub w1: DMatrix<f64>,
    pub b1: DMatrix<f64>,
    pub w2: DMatrix<f64>,
    pub b2: DMatrix<f64>,
}
pub struct Two_layer_network {
    pub w1: Rc<RefCell<DMatrix<f64>>>,
    pub b1: Rc<RefCell<DMatrix<f64>>>,
    pub w2: Rc<RefCell<DMatrix<f64>>>,
    pub b2: Rc<RefCell<DMatrix<f64>>>,
}
impl Two_layer_network {
    pub fn predict(&self, x: &DMatrix<f64>) -> DMatrix<f64> {
        let w1 = &*self.w1.borrow();
        let b1 = &*self.b1.borrow();
        let w2 = &*self.w2.borrow();
        let b2 = &*self.b2.borrow();
        let a1 = nural::dot(x, w1) + b1;
        let z1 = nural::sigmoid(&a1);
        let a2 = nural::dot(&z1, w2) + b2;
        let y = nural::softmax(&a2);
        y
    }
    pub fn loss(
        &self,
        param: &DMatrix<f64>,
        x: &DMatrix<f64>,
        t: &DMatrix<f64>,
        patern: &str,
    ) -> f64 {
        match patern {
            "w1" => {
                let y = predict(
                    param,
                    &*self.b1.borrow(),
                    &*self.w2.borrow(),
                    &*self.b2.borrow(),
                    x,
                );
                lossfunc::cross_entropy(&y, t)
            }
            "b1" => {
                let y = predict(
                    &*self.b1.borrow(),
                    param,
                    &*self.w2.borrow(),
                    &*self.b2.borrow(),
                    x,
                );
                lossfunc::cross_entropy(&y, t)
            }
            "w2" => {
                let y = predict(
                    &*self.b1.borrow(),
                    &*self.b1.borrow(),
                    param,
                    &*self.b2.borrow(),
                    x,
                );
                lossfunc::cross_entropy(&y, t)
            }
            "b2" => {
                let y = predict(
                    &*self.b1.borrow(),
                    &*self.b1.borrow(),
                    &*self.w2.borrow(),
                    param,
                    x,
                );
                lossfunc::cross_entropy(&y, t)
            }
            _ => {
                let y = self.predict(x);
                lossfunc::cross_entropy(&y, t)
            }
        }
    }
    pub fn numerical_gradient(&mut self, x: &DMatrix<f64>, t: &DMatrix<f64>) -> grad {

        let grads = grad {
            w1: gradient::numerical_gradient(loss_w, &self, &self.w1, x, t, &"w1"),
            b1: gradient::numerical_gradient(loss_w, &self, &self.b1, x, t, &"b1"),
            w2: gradient::numerical_gradient(loss_w, &self, &self.w2, x, t, &"w2"),
            b2: gradient::numerical_gradient(loss_w, &self, &self.b2, x, t, &"b2"),
        };
        grads
    }
    pub fn gradient(&self, x: &DMatrix<f64>, t: &DMatrix<f64>) -> grad {
        let (w1, b1, w2, b2) = (
            &*self.w1.borrow(),
            &*self.b1.borrow(),
            &*self.w2.borrow(),
            &*self.b2.borrow(),
        );
        let batch_num = x.shape().0;
        // forward
        let a1 = nural::dot(x, w1) + b1;
        let z1 = nural::sigmoid(&a1);
        let a2 = nural::dot(&z1, w2) + b2;
        let y = nural::softmax(&a2);
        //backword
        let dy = (y - t).map(|i| i / batch_num as f64);
        let da1 = nural::dot(&dy, &w2.transpose());
        let dz1 = nural::sigmoid_grad(&a1) * da1;
        let grads = grad {
            w1: nural::dot(&x.transpose(), &dz1),
            b1: nural::axisZerosum(&dz1),
            w2: nural::dot(&z1.transpose(), &dy),
            b2: nural::axisZerosum(&dy),
        };
        grads
    }
}

pub fn loss_w(
    param: &DMatrix<f64>,
    x: &DMatrix<f64>,
    t: &DMatrix<f64>,
    two: &Two_layer_network,
    patern: &'static str,
) -> f64 {
    two.loss(param, x, t, &patern)
}
pub fn learn(late: &f64, matrix: &DMatrix<f64>) -> DMatrix<f64> {
    matrix.map(|i| i * late)
}
pub fn predict(
    w1: &DMatrix<f64>,
    b1: &DMatrix<f64>,
    w2: &DMatrix<f64>,
    b2: &DMatrix<f64>,
    x: &DMatrix<f64>,
) -> DMatrix<f64> {
    println!("{:?}", x.shape());
    println!("{:?}", w1.shape());
    println!("{:?}", b1.shape());
    let a1 = nural::dot(x, w1) + b1;
    println!("{:?}", a1.shape());
    let z1 = nural::sigmoid(&a1);
    println!("{:?}", z1.shape());
    let a2 = nural::dot(&z1, w2) + b2;
    println!("{:?}", a2.shape());
    let y = nural::softmax(&a2);
    y
}
