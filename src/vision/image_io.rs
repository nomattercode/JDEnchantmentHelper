use std::fs;
use std::path::Path;

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

pub fn load_image(path: impl AsRef<Path>) -> Result<Mat, String> {
    let path = path.as_ref();
    let bytes = fs::read(path)
        .map_err(|error| format!("Не удалось прочитать '{}': {error}", path.display()))?;
    let image = decode_image(&bytes)
        .map_err(|error| format!("OpenCV не смог декодировать '{}': {error}", path.display()))?;
    if image.empty() {
        return Err(format!("Файл '{}' не содержит декодируемого изображения", path.display()));
    }
    Ok(image)
}

/// Encode with OpenCV and write with std::fs to preserve Unicode paths.
pub fn save_png(path: impl AsRef<Path>, image: &Mat) -> Result<(), String> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Не удалось создать '{}': {error}", parent.display()))?;
    }
    let mut encoded = Vector::<u8>::new();
    imgcodecs::imencode(".png", image, &mut encoded, &Vector::<i32>::new())
        .map_err(|error| format!("OpenCV не смог сохранить '{}': {error}", path.display()))?;
    fs::write(path, encoded.as_slice())
        .map_err(|error| format!("Не удалось записать '{}': {error}", path.display()))
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
