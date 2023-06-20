use wasm_bindgen::prelude::*;
use web_sys::{HtmlCanvasElement, CanvasRenderingContext2d};

#[wasm_bindgen(start)]
pub fn main() -> Result<(), JsValue> {
    match get_canvas_element() {
        Ok(canvas) => {
            let ctx = canvas.get_context("2d")?.unwrap().dyn_into::<CanvasRenderingContext2d>()?;

            ctx.set_fill_style(&JsValue::from_str("red"));
            ctx.fill_rect(10.0, 10.0, 50.0, 50.0);

            Ok(())
        },
        Err(err) => Err(err)
    }
}

fn get_canvas_element() -> Result<HtmlCanvasElement, JsValue> {
    let window = web_sys::window().ok_or(JsValue::NULL)?;
    let document = window.document().ok_or(JsValue::NULL)?;
    let canvas = document.get_element_by_id("render-target").ok_or(JsValue::NULL)?.dyn_into::<HtmlCanvasElement>()?;
    Ok(canvas)
}