use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size, Vector},
    imgproc,
    prelude::*,
};

use crate::vision::image_io::decode_image;

const SUCCESS_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/ok_success.png");
const FAILURE_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/ok_failure.png");

#[derive(Clone, Copy, Debug, Default)]
pub struct Region {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Region {
    pub fn is_configured(self) -> bool {
        self.width > 0 && self.height > 0
    }
}

#[derive(Clone, Debug)]
pub struct EnchantDetectorConfig {
    pub confidence_threshold: f64,
    pub scale_min: f64,
    pub scale_max: f64,
    pub scale_step: f64,
    pub search_region: Option<Region>,
    /// A zero-sized region means automatic detection over the full frame.
    /// Set a rectangle in the UI to restrict detection to a known button area.
    pub start_button_region: Region,
    pub min_blue_ratio: f64,
}

impl Default for EnchantDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.84,
            scale_min: 0.80,
            scale_max: 1.25,
            scale_step: 0.05,
            search_region: None,
            start_button_region: Region::default(),
            min_blue_ratio: 0.22,
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
pub struct StartButtonState {
    pub active: bool,
    pub blue_ratio: f64,
    pub rect: Rect,
}

#[derive(Debug)]
pub struct EnchantAnalysis {
    pub ok_button: MatchResult,
    pub start_button: Option<StartButtonState>,
    pub overlay: Mat,
    pub log: Vec<String>,
}

fn grayscale(image: &Mat) -> opencv::Result<Mat> {
    let mut gray = Mat::default();
    imgproc::cvt_color(
        image,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;
    Ok(gray)
}

fn match_template(
    source: &Mat,
    template: &Mat,
    name: &str,
    cfg: &EnchantDetectorConfig,
) -> opencv::Result<MatchResult> {
    let mut best_score = -1.0_f64;
    let mut best_rect = None;
    let mut best_scale = 1.0;
    let step = cfg.scale_step.max(0.01);
    let mut scale = cfg.scale_min.max(0.1);

    while scale <= cfg.scale_max.max(cfg.scale_min) + 0.0001 {
        let width = (f64::from(template.cols()) * scale).round() as i32;
        let height = (f64::from(template.rows()) * scale).round() as i32;
        if width >= 8 && height >= 8 && width <= source.cols() && height <= source.rows() {
            let mut resized = Mat::default();
            imgproc::resize(
                template,
                &mut resized,
                Size::new(width, height),
                0.0,
                0.0,
                imgproc::INTER_AREA,
            )?;
            let mut response = Mat::default();
            imgproc::match_template(
                source,
                &resized,
                &mut response,
                imgproc::TM_CCOEFF_NORMED,
                &Mat::default(),
            )?;

            let (mut min_value, mut max_value) = (0.0, 0.0);
            let (mut min_point, mut max_point) = (Point::default(), Point::default());
            core::min_max_loc(
                &response,
                Some(&mut min_value),
                Some(&mut max_value),
                Some(&mut min_point),
                Some(&mut max_point),
                &core::no_array(),
            )?;
            if max_value > best_score {
                best_score = max_value;
                best_rect = Some(Rect::new(max_point.x, max_point.y, width, height));
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

fn clip_region(region: Region, frame: &Mat) -> Option<Rect> {
    let x1 = region.x.max(0).min(frame.cols());
    let y1 = region.y.max(0).min(frame.rows());
    let x2 = (region.x + region.width).max(0).min(frame.cols());
    let y2 = (region.y + region.height).max(0).min(frame.rows());

    if x2 <= x1 || y2 <= y1 {
        None
    } else {
        Some(Rect::new(x1, y1, x2 - x1, y2 - y1))
    }
}

fn blue_mask(image: &Mat) -> opencv::Result<Mat> {
    let mut hsv = Mat::default();
    imgproc::cvt_color(
        image,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // OpenCV hue is 0..179. This range targets saturated blue button fills,
    // while excluding neutral gray buttons and most cyan/green UI elements.
    let lower = Scalar::new(92.0, 85.0, 65.0, 0.0);
    let upper = Scalar::new(140.0, 255.0, 255.0, 0.0);
    let mut mask = Mat::default();
    core::in_range(&hsv, &lower, &upper, &mut mask)?;
    Ok(mask)
}

fn detect_start_button(
    frame: &Mat,
    cfg: &EnchantDetectorConfig,
) -> opencv::Result<Option<StartButtonState>> {
    let configured_region = cfg.start_button_region.is_configured();

    if configured_region {
        let Some(rect) = clip_region(cfg.start_button_region, frame) else {
            return Ok(None);
        };
        if rect.width != cfg.start_button_region.width || rect.height != cfg.start_button_region.height {
            return Ok(None);
        }

        let crop = Mat::roi(frame, rect)?;
        let mask = blue_mask(&crop)?;
        let ratio = f64::from(core::count_non_zero(&mask)?)
            / f64::from((rect.width * rect.height).max(1));
        return Ok(Some(StartButtonState {
            active: ratio >= cfg.min_blue_ratio,
            blue_ratio: ratio,
            rect,
        }));
    }

    // Automatic mode: locate blue, button-shaped connected regions anywhere
    // in the captured window. The aspect/size filters reject small blue icons
    // and long status bars. A manual ROI can be set for unusual game layouts.
    let mask = blue_mask(frame)?;
    let mut contours = Vector::<Vector<Point>>::new();
    imgproc::find_contours(
        &mask,
        &mut contours,
        imgproc::RETR_EXTERNAL,
        imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut best: Option<(Rect, f64, f64)> = None;
    for index in 0..contours.len() {
        let contour = contours.get(index)?;
        let rect = imgproc::bounding_rect(&contour)?;
        if rect.width < 36 || rect.height < 12 || rect.width > 700 || rect.height > 140 {
            continue;
        }

        let aspect = f64::from(rect.width) / f64::from(rect.height.max(1));
        if !(1.25..=7.5).contains(&aspect) {
            continue;
        }

        let roi = Mat::roi(&mask, rect)?;
        let blue_pixels = f64::from(core::count_non_zero(&roi)?);
        let area = f64::from(rect.width * rect.height);
        let ratio = blue_pixels / area.max(1.0);
        if ratio < cfg.min_blue_ratio {
            continue;
        }

        let score = ratio * area.sqrt();
        if best.as_ref().is_none_or(|(_, _, best_score)| score > *best_score) {
            best = Some((rect, ratio, score));
        }
    }

    Ok(best.map(|(rect, blue_ratio, _)| StartButtonState {
        active: true,
        blue_ratio,
        rect,
    }))
}

pub fn analyze_enchant_window(
    frame: &Mat,
    cfg: &EnchantDetectorConfig,
) -> opencv::Result<EnchantAnalysis> {
    if frame.empty() {
        return Err(opencv::Error::new(core::StsBadArg, "Входной кадр пуст"));
    }

    let (search, offset) = if let Some(region) = cfg.search_region {
        let rect = clip_region(region, frame).ok_or_else(|| {
            opencv::Error::new(core::StsBadArg, "Область поиска OK вне кадра")
        })?;
        (Mat::roi(frame, rect)?.try_clone()?, Point::new(rect.x, rect.y))
    } else {
        (frame.try_clone()?, Point::new(0, 0))
    };

    let source = grayscale(&search)?;
    let success = grayscale(&decode_image(SUCCESS_TEMPLATE)?)?;
    let failure = grayscale(&decode_image(FAILURE_TEMPLATE)?)?;

    let mut best = match_template(&source, &success, "ok_success", cfg)?;
    let failure_match = match_template(&source, &failure, "ok_failure", cfg)?;
    if failure_match.score > best.score {
        best = failure_match;
    }
    if let Some(rect) = &mut best.rect {
        rect.x += offset.x;
        rect.y += offset.y;
    }

    let start_button = detect_start_button(frame, cfg)?;
    let mut overlay = frame.try_clone()?;
    let mut log = vec![format!(
        "[enchant] frame={}x{} OK threshold={:.3} scale={:.2}..{:.2} step={:.2}",
        frame.cols(),
        frame.rows(),
        cfg.confidence_threshold,
        cfg.scale_min,
        cfg.scale_max,
        cfg.scale_step
    )];

    if let Some(rect) = best.rect {
        let color = if best.found {
            Scalar::new(0.0, 220.0, 0.0, 0.0)
        } else {
            Scalar::new(0.0, 180.0, 255.0, 0.0)
        };
        imgproc::rectangle(&mut overlay, rect, color, 2, imgproc::LINE_8, 0)?;
        imgproc::put_text(
            &mut overlay,
            &format!("{} {:.3}", best.template_name, best.score),
            Point::new(rect.x, (rect.y - 5).max(14)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            imgproc::LINE_AA,
            false,
        )?;
        log.push(format!(
            "[enchant] best={} score={:.4} threshold={:.3} scale={:.2} rect=({}, {}, {}, {}) found={}",
            best.template_name,
            best.score,
            cfg.confidence_threshold,
            best.scale,
            rect.x,
            rect.y,
            rect.width,
            rect.height,
            best.found
        ));
    } else {
        log.push(format!(
            "[enchant] no OK match; best score={:.4}",
            best.score
        ));
    }

    if let Some(button) = &start_button {
        let color = if button.active {
            Scalar::new(0.0, 220.0, 0.0, 0.0)
        } else {
            Scalar::new(0.0, 0.0, 220.0, 0.0)
        };
        imgproc::rectangle(&mut overlay, button.rect, color, 2, imgproc::LINE_8, 0)?;
        imgproc::put_text(
            &mut overlay,
            &format!("START blue={:.2}", button.blue_ratio),
            Point::new(button.rect.x, (button.rect.y - 5).max(14)),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            imgproc::LINE_AA,
            false,
        )?;
        log.push(format!(
            "[enchant] start blue_ratio={:.3} threshold={:.3} active={} rect=({}, {}, {}, {})",
            button.blue_ratio,
            cfg.min_blue_ratio,
            button.active,
            button.rect.x,
            button.rect.y,
            button.rect.width,
            button.rect.height
        ));
    } else {
        log.push("[enchant] active blue Start button not found".into());
    }

    Ok(EnchantAnalysis {
        ok_button: best,
        start_button,
        overlay,
        log,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_ok_template_matches_itself() {
        let frame = decode_image(SUCCESS_TEMPLATE).unwrap();
        let cfg = EnchantDetectorConfig {
            confidence_threshold: 0.95,
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = analyze_enchant_window(&frame, &cfg).unwrap();
        assert!(result.ok_button.found, "score={}", result.ok_button.score);
        assert!(result.ok_button.score > 0.99);
    }

    #[test]
    fn failure_ok_template_matches_itself() {
        let frame = decode_image(FAILURE_TEMPLATE).unwrap();
        let cfg = EnchantDetectorConfig {
            confidence_threshold: 0.95,
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = analyze_enchant_window(&frame, &cfg).unwrap();
        assert!(result.ok_button.found, "score={}", result.ok_button.score);
        assert!(result.ok_button.score > 0.99);
    }

    #[test]
    fn blue_start_button_is_detected_automatically() {
        let mut frame = Mat::new_rows_cols_with_default(
            240,
            320,
            core::CV_8UC3,
            Scalar::new(70.0, 70.0, 70.0, 0.0),
        )
        .unwrap();
        imgproc::rectangle(
            &mut frame,
            Rect::new(90, 150, 130, 32),
            Scalar::new(220.0, 110.0, 15.0, 0.0),
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )
        .unwrap();

        let result = detect_start_button(&frame, &EnchantDetectorConfig::default())
            .unwrap()
            .expect("blue Start button should be found");
        assert!(result.active);
        assert!(result.blue_ratio >= 0.95);
        assert!(result.rect.x >= 90 && result.rect.x <= 92);
    }

    #[test]
    fn gray_start_button_is_not_triggered() {
        let mut frame = Mat::new_rows_cols_with_default(
            240,
            320,
            core::CV_8UC3,
            Scalar::new(40.0, 40.0, 40.0, 0.0),
        )
        .unwrap();
        imgproc::rectangle(
            &mut frame,
            Rect::new(90, 150, 130, 32),
            Scalar::new(140.0, 140.0, 140.0, 0.0),
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )
        .unwrap();

        let result = detect_start_button(&frame, &EnchantDetectorConfig::default()).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn configured_start_roi_reports_blue_ratio() {
        let mut frame = Mat::new_rows_cols_with_default(
            240,
            320,
            core::CV_8UC3,
            Scalar::new(60.0, 60.0, 60.0, 0.0),
        )
        .unwrap();
        imgproc::rectangle(
            &mut frame,
            Rect::new(100, 160, 100, 28),
            Scalar::new(220.0, 110.0, 15.0, 0.0),
            imgproc::FILLED,
            imgproc::LINE_8,
            0,
        )
        .unwrap();

        let cfg = EnchantDetectorConfig {
            start_button_region: Region {
                x: 100,
                y: 160,
                width: 100,
                height: 28,
            },
            ..Default::default()
        };
        let result = detect_start_button(&frame, &cfg).unwrap().unwrap();
        assert!(result.active);
        assert!(result.blue_ratio > 0.95);
    }
}
