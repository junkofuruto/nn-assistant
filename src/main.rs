use rand::Rng;

fn generate_model_weigth() -> f64 {
    return rand::thread_rng().gen_range(-1.0..=1.0);
}

fn model_cost(data: &[[f64; 2]; 4], model: &f64, bias: &f64) -> f64 {
    let mut result: f64 = 0.0;
    let mut x: f64;
    let mut y: f64;
    let mut d: f64;
    for el in data {
        x = el[0];
        y = x * model + bias;
        d = y - el[1];
        result += d * d;
    }
    result /= data.len() as f64;
    return result; 
}

fn main() {
    let data:[[f64; 2]; 4] = [
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
    ];
    let mut weigth = generate_model_weigth() * 10.0;
    let mut bias = generate_model_weigth() * 5.0;
    let eps = 1e-3;
    println!("W: {0}\nB: {1}", &weigth, &bias);
    
    while model_cost(&data, &weigth, &bias) > eps {
        let cost = model_cost(&data, &weigth, &bias);
        let dw = (model_cost(&data, &(weigth - &eps), &bias) - cost) / eps;
        let db = (model_cost(&data, &weigth, &(bias  - &eps)) - cost) / eps;
        weigth += dw * eps;
        bias += db * eps;
        println!("| C: {0}", model_cost(&data, &weigth, &bias));
    }
    
    println!("W: {0}\nB: {1}", &weigth, &bias);
    println!("\n+-- X --+- EXP -+- ACT -+");
    for el in data {
        println!("| {0:.3} | {1:.3} | {2:.3} |", el[0], el[1], el[0] * &weigth);
    }
    println!("+-------+-------+-------+");
}