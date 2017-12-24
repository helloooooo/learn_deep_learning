
extern crate nalgebra;
use nalgebra::core::DMatrix;
use std::rc::Rc;
use std::cell::{RefCell, Ref, RefMut};
use two_layer_net;

pub fn numerical_gradient<
    F: Fn(&DMatrix<f64>,
       &DMatrix<f64>,
       &DMatrix<f64>,
       &two_layer_net::Two_layer_network,
       &'static str,)
       -> f64,
>(
    f: F,
    two: &two_layer_net::Two_layer_network,
    param: &Rc<RefCell<DMatrix<f64>>>,
    x: &DMatrix<f64>,
    t: &DMatrix<f64>,
    patern:&'static str,
) -> DMatrix<f64> {
    let h = 1.0e-4;
    let mut grad = DMatrix::<f64>::from_element(param.borrow().nrows(), param.borrow().ncols(), 0.0);
    let length = param.borrow().len();
    let mut cp_param = param.borrow_mut() ;
    for i in 0..length{
        let tmp_val = cp_param[i];
        cp_param[i] = tmp_val + h;
      
        let fxh1 = f(&cp_param, x, t, two,&patern);
        cp_param[i] = tmp_val - h;
        let fxh2 = f(&cp_param, x, t, two,&patern);

        grad[i] = (fxh1 - fxh2) / (2.0 * h);
        cp_param[i] = tmp_val;
    }
    grad
}
pub fn function_2(x: &mut DMatrix<f64>) -> f64 {
    x.iter().map(|x| x.powi(2)).sum()
}


#[test]
fn p102() {
    let t = &mut DMatrix::<f64>::from_iterator(
        10,
        1,
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            .iter()
            .cloned(),
    );
    assert_eq!(385.0, function_2(t));
}
