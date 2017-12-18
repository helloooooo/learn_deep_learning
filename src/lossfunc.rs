extern crate nalgebra;
use nalgebra::core::{DMatrix};

pub fn cross_entropy(y: DMatrix<f64>,t:DMatrix<f64>) -> f64{
    let delta = 1.0e-7;
    -1.0 * y.iter().zip(t.iter()).fold(0.0,|amount,(_y,_t)|{
        amount + _t * (_y + delta).ln()
    })
}
pub fn mean_squared_error(y:DMatrix<f64>,t:DMatrix<f64>) -> f64{
    0.5 *  y.iter().zip(t.iter()).fold(0.0, |amount,(_y,_t)|{
        amount + (_y - _t).powi(2)
    })
}

#[test]
fn p91(){
    let t = DMatrix::<f64>::from_iterator(10,1,[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0].iter().cloned());
    let y = DMatrix::<f64>::from_iterator(10,1,[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0].iter().cloned());
    assert_eq!(0.51082545709933802,cross_entropy(y,t) );
}
#[test]
fn p89(){
    let t = DMatrix::<f64>::from_iterator(10,1,[0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0].iter().cloned());
    let y = DMatrix::<f64>::from_iterator(10,1,[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0].iter().cloned());
    assert_eq!(0.097500000000000031,mean_squared_error(y,t) );
}