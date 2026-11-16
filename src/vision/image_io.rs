use opencv::{
    core::{self, Mat, Vector},
    imgcodecs,
    imgproc,
    prelude::*,
};

/// Decode bytes rather than passing a path to OpenCV. This supports Unicode
/// Windows paths, including Cyrillic folder and file names.
pub fn decode_image(bytes: &[u8]) -> opencv::Result<Mat> {
    let encoded = Vector::<u8>::from_slice(bytes);
    imgcodecs::imdecode(&encoded, imgcodecs::IMREAD_COLOR)
}

pub fn mat_to_rgb(image: &Mat) -> opencv::Result<(usize, usize, Vec<u8>)> {
    let mut rgb = Mat::default();
    match image.channels() {
        1 => imgproc::cvt_color(image, &mut rgb, imgproc::COLOR_GRAY2RGB, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?,
        3 => imgproc::cvt_color(image, &mut rgb, imgproc::COLOR_BGR2RGB, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?,
        4 => imgproc::cvt_color(image, &mut rgb, imgproc::COLOR_BGRA2RGB, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?,
        channels => {
            return Err(opencv::Error::new(
                core::StsUnsupportedFormat,
                format!("Неподдерживаемое число каналов: {channels}"),
            ));
        }
    }
    let bytes = rgb.data_bytes()?.to_vec();
    Ok((rgb.cols() as usize, rgb.rows() as usize, bytes))
}
