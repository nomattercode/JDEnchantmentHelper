use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    imgproc,
    prelude::*,
};

use crate::vision::image_io::decode_image;

const SUCCESS_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/ok_success.png");
const FAILURE_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/ok_failure.png");

#[derive(Clone, Copy, Debug, Default)]
pub struct Region { pub x: i32, pub y: i32, pub width: i32, pub height: i32 }

#[derive(Clone, Debug)]
pub struct EnchantDetectorConfig {
    pub confidence_threshold: f64,
    pub scale_min: f64,
    pub scale_max: f64,
    pub scale_step: f64,
    pub search_region: Option<Region>,
    pub check_start_button: bool,
    pub start_button_region: Region,
    pub active_saturation_threshold: f64,
}

impl Default for EnchantDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.84, scale_min: 0.80, scale_max: 1.25, scale_step: 0.05,
            search_region: None, check_start_button: false,
            // Approximate only for the 324x220 dialog crop; recalibrate on a full client capture.
            start_button_region: Region { x: 43, y: 158, width: 72, height: 28 },
            active_saturation_threshold: 0.18,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatchResult {
    pub found: bool,
    pub score: f64,
    pub rect: Option<Rect>,
    pub template_name: String,
    pub scale: f64,
}

#[derive(Clone, Debug)]
pub struct StartButtonState { pub active: bool, pub saturation: f64, pub rect: Rect }

#[derive(Debug)]
pub struct EnchantAnalysis {
    pub ok_button: MatchResult,
    pub start_button: Option<StartButtonState>,
    pub overlay: Mat,
    pub log: Vec<String>,
}

fn grayscale(image: &Mat) -> opencv::Result<Mat> {
    let mut gray = Mat::default();
    imgproc::cvt_color(image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
    Ok(gray)
}

fn match_template(source: &Mat, template: &Mat, name: &str, cfg: &EnchantDetectorConfig) -> opencv::Result<MatchResult> {
    let mut best_score = -1.0_f64;
    let mut best_rect = None;
    let mut best_scale = 1.0;
    let step = cfg.scale_step.max(0.01);
    let mut scale = cfg.scale_min.max(0.1);
    while scale <= cfg.scale_max.max(cfg.scale_min) + 0.0001 {
        let w = (f64::from(template.cols()) * scale).round() as i32;
        let h = (f64::from(template.rows()) * scale).round() as i32;
        if w >= 8 && h >= 8 && w <= source.cols() && h <= source.rows() {
            let mut resized = Mat::default();
            imgproc::resize(template, &mut resized, Size::new(w, h), 0.0, 0.0, imgproc::INTER_AREA)?;
            let mut response = Mat::default();
            imgproc::match_template(source, &resized, &mut response, imgproc::TM_CCOEFF_NORMED, &Mat::default())?;
            let (mut min_v, mut max_v) = (0.0, 0.0);
            let (mut min_p, mut max_p) = (Point::default(), Point::default());
            core::min_max_loc(&response, Some(&mut min_v), Some(&mut max_v), Some(&mut min_p), Some(&mut max_p), &core::no_array())?;
            if max_v > best_score {
                best_score = max_v;
                best_rect = Some(Rect::new(max_p.x, max_p.y, w, h));
                best_scale = scale;
            }
        }
        scale += step;
    }
    Ok(MatchResult {
        found: best_score >= cfg.confidence_threshold,
        score: best_score.max(0.0),
        rect: best_rect,
        template_name: name.into(),
        scale: best_scale,
    })
}

fn clip_region(r: Region, frame: &Mat) -> Option<Rect> {
    let x1 = r.x.max(0).min(frame.cols());
    let y1 = r.y.max(0).min(frame.rows());
    let x2 = (r.x + r.width).max(0).min(frame.cols());
    let y2 = (r.y + r.height).max(0).min(frame.rows());
    if x2 <= x1 || y2 <= y1 { None } else { Some(Rect::new(x1, y1, x2 - x1, y2 - y1)) }
}

fn detect_start_button(frame: &Mat, cfg: &EnchantDetectorConfig) -> opencv::Result<Option<StartButtonState>> {
    let Some(rect) = clip_region(cfg.start_button_region, frame) else { return Ok(None); };
    if rect.width != cfg.start_button_region.width || rect.height != cfg.start_button_region.height {
        return Ok(None);
    }
    let crop = Mat::roi(frame, rect)?;
    let mut hsv = Mat::default();
    imgproc::cvt_color(&crop, &mut hsv, imgproc::COLOR_BGR2HSV, 0)?;
    let mut channels = Vector::<Mat>::new();
    core::split(&hsv, &mut channels)?;
    let sat = channels.get(1)?;
    let value = core::mean(&sat, &core::no_array())?[0] / 255.0;
    Ok(Some(StartButtonState { active: value >= cfg.active_saturation_threshold, saturation: value, rect }))
}

pub fn analyze_enchant_window(frame: &Mat, cfg: &EnchantDetectorConfig) -> opencv::Result<EnchantAnalysis> {
    if frame.empty() { return Err(opencv::Error::new(core::StsBadArg, "Входной кадр пуст")); }
    let (search, offset) = if let Some(region) = cfg.search_region {
        let rect = clip_region(region, frame).ok_or_else(|| opencv::Error::new(core::StsBadArg, "Область поиска OK вне кадра"))?;
        (Mat::roi(frame, rect)?.try_clone()?, Point::new(rect.x, rect.y))
    } else {
        (frame.try_clone()?, Point::new(0, 0))
    };
    let source = grayscale(&search)?;
    let success = grayscale(&decode_image(SUCCESS_TEMPLATE)?)?;
    let failure = grayscale(&decode_image(FAILURE_TEMPLATE)?)?;
    let mut best = match_template(&source, &success, "ok_success", cfg)?;
    let failure_match = match_template(&source, &failure, "ok_failure", cfg)?;
    if failure_match.score > best.score { best = failure_match; }
    if let Some(rect) = &mut best.rect { rect.x += offset.x; rect.y += offset.y; }

    let start_button = if cfg.check_start_button { detect_start_button(frame, cfg)? } else { None };
    let mut overlay = frame.try_clone()?;
    let mut log = vec![format!(
        "[enchant] frame={}x{} threshold={:.3} scale={:.2}..{:.2} step={:.2}",
        frame.cols(), frame.rows(), cfg.confidence_threshold, cfg.scale_min, cfg.scale_max, cfg.scale_step
    )];
    if let Some(rect) = best.rect {
        let color = if best.found { Scalar::new(0.0, 220.0, 0.0, 0.0) } else { Scalar::new(0.0, 180.0, 255.0, 0.0) };
        imgproc::rectangle(&mut overlay, rect, color, 2, imgproc::LINE_8, 0)?;
        imgproc::put_text(&mut overlay, &format!("{} {:.3}", best.template_name, best.score),
            Point::new(rect.x, (rect.y - 5).max(14)), imgproc::FONT_HERSHEY_SIMPLEX,
            0.45, color, 1, imgproc::LINE_AA, false)?;
        log.push(format!(
            "[enchant] best={} score={:.4} threshold={:.3} scale={:.2} rect=({}, {}, {}, {}) found={}",
            best.template_name, best.score, cfg.confidence_threshold, best.scale,
            rect.x, rect.y, rect.width, rect.height, best.found
        ));
    } else {
        log.push(format!("[enchant] no match; best score={:.4}", best.score));
    }
    if let Some(state) = &start_button {
        let color = if state.active { Scalar::new(0.0, 220.0, 0.0, 0.0) } else { Scalar::new(0.0, 0.0, 220.0, 0.0) };
        imgproc::rectangle(&mut overlay, state.rect, color, 2, imgproc::LINE_8, 0)?;
        log.push(format!("[enchant] insert_button saturation={:.3} threshold={:.3} active={}",
            state.saturation, cfg.active_saturation_threshold, state.active));
    } else if cfg.check_start_button {
        log.push("[enchant] Insert ROI outside frame or clipped; recalibrate coordinates".into());
    }
    Ok(EnchantAnalysis { ok_button: best, start_button, overlay, log })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_ok_template_matches_itself() {
        let frame = decode_image(SUCCESS_TEMPLATE).unwrap();
        let cfg = EnchantDetectorConfig { confidence_threshold: 0.95, scale_min: 1.0, scale_max: 1.0, scale_step: 0.05, ..Default::default() };
        let result = analyze_enchant_window(&frame, &cfg).unwrap();
        assert!(result.ok_button.found, "score={}", result.ok_button.score);
        assert!(result.ok_button.score > 0.99);
    }

    #[test]
    fn failure_ok_template_matches_itself() {
        let frame = decode_image(FAILURE_TEMPLATE).unwrap();
        let cfg = EnchantDetectorConfig { confidence_threshold: 0.95, scale_min: 1.0, scale_max: 1.0, scale_step: 0.05, ..Default::default() };
        let result = analyze_enchant_window(&frame, &cfg).unwrap();
        assert!(result.ok_button.found, "score={}", result.ok_button.score);
        assert!(result.ok_button.score > 0.99);
    }
}
