extern crate nalgebra;
use nalgebra::core::DMatrix;


pub fn numerical_gradient<F>(f: F, x: &mut DMatrix<f64>) -> DMatrix<f64>
where
    F: Fn(&mut DMatrix<f64>) -> f64,
{
    let h = 1.0e-4;
    let mut grad = DMatrix::<f64>::from_element(x.nrows(), x.ncols(), 0.0);
    for i in 0..x.len() {
        let tmp_val = x[i];
        x[i] = tmp_val + h;
        let fxh1 = f(x);
        x[i] = tmp_val - h;
        let fxh2 = f(x);

        grad[i] = (fxh1 - fxh2) / (2.0 * h);
        x[i] = tmp_val;
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
#[test]
fn p104() {
    let t = &mut DMatrix::<f64>::from_iterator(2, 1, [3.0, 4.0].iter().cloned());
    let ans =
        DMatrix::<f64>::from_iterator(2, 1, [6.00000000000378, 7.999999999999119].iter().cloned());
    assert_eq!(ans, numerical_gradient(function_2, t));
}
