use rand::Rng;

fn generate_model_weigth() -> f64 {
    return rand::thread_rng().gen_range(-1.0..=1.0);
}

fn model_cost(data: &[[f64; 3]; 4], w1: &f64, w2: &f64) -> f64 {
    let mut result: f64 = 0.0;
    for el in data {
        let x1: &f64 = &el[0];
        let x2: &f64 = &el[1];
        let y: f64 = x1 * w1 + x2 * w2;
        let d: f64 = y - el[2];
        result += d * d;
    }
    result /= data.len() as f64;
    return result; 
}

fn main() {
    let eps:f64 = 1e-3;
    let rate: f64 = 1e-2;
    let data:[[f64; 3]; 4] = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let mut w1: f64 = generate_model_weigth();
    let mut w2: f64 = generate_model_weigth();
    let mut cost: f64 = model_cost(&data, &w1, &w2);
    
    while cost > 0.1 {
        cost = model_cost(&data, &w1, &w2);
        let dw1: f64 = (model_cost(&data, &(w1 + eps), &w2) - cost) / eps;
        let dw2: f64 = (model_cost(&data, &w1, &(w2 + eps)) - cost) / eps;
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        println!("| W1: {0:.5}\tW2: {1:.5}\tC: {2:.5}", w1, w2, cost);
    }

    for el in data {
        println!("{0} {1} Ye:{2} Ya:{3:.0}", el[0], el[1], el[2], (el[0] * w1 + el[1] * w2));
    }
}
