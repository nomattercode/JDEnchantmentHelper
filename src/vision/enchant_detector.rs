use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size},
    imgproc,
    prelude::*,
};

use serde::{Deserialize, Serialize};
use crate::vision::image_io::decode_image;

const OK_BUTTON_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/ok_button.png");
const OK_BUTTON_SCALE_MIN: f64 = 0.65;
const OK_BUTTON_SCALE_MAX: f64 = 1.45;
const OK_BUTTON_SCALE_STEP: f64 = 0.05;
const START_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/start_button_active.png");
const ENCHANT_HEADER_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/enchant_header.png");

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct EnchantDetectorConfig {
    pub confidence_threshold: f64,
    pub header_threshold: f64,
    pub scale_min: f64,
    pub scale_max: f64,
    pub scale_step: f64,
    pub search_region: Option<Region>,
    /// A zero-sized region means automatic detection over the full frame.
    /// Set a rectangle in the UI to restrict detection to a known button area.
    pub start_button_region: Region,
    pub start_template_threshold: f64,
    pub min_blue_ratio: f64,
}

impl Default for EnchantDetectorConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.78,
            header_threshold: 0.64,
            scale_min: 0.80,
            scale_max: 1.25,
            scale_step: 0.05,
            search_region: None,
            start_button_region: Region::default(),
            start_template_threshold: 0.68,
            min_blue_ratio: 0.45,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MatchResult {
    pub found: bool,
    pub score: f64,
    pub rect: Option<Rect>,
    pub scale: f64,
}

#[derive(Clone, Debug)]
pub struct StartButtonState {
    pub active: bool,
    pub rect: Rect,
}

#[derive(Debug)]
pub struct EnchantAnalysis {
    pub dialog_window: Option<Rect>,
    pub ok_button: MatchResult,
    pub start_button: Option<StartButtonState>,
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
    cfg: &EnchantDetectorConfig,
    threshold: f64,
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
                if scale > 1.0 { imgproc::INTER_LINEAR } else { imgproc::INTER_AREA },
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
        found: best_score >= threshold,
        score: best_score.max(0.0),
        rect: best_rect,
        scale: best_scale,
    })
}

fn clip_region(region: Region, frame: &Mat) -> Option<Rect> {
    let x1 = region.x.max(0).min(frame.cols());
    let y1 = region.y.max(0).min(frame.rows());
    let x2 = region.x.saturating_add(region.width).max(0).min(frame.cols());
    let y2 = region.y.saturating_add(region.height).max(0).min(frame.rows());

    if x2 <= x1 || y2 <= y1 {
        None
    } else {
        Some(Rect::new(x1, y1, x2 - x1, y2 - y1))
    }
}

fn intersect_rects(left: Rect, right: Rect) -> Option<Rect> {
    let x1 = left.x.max(right.x);
    let y1 = left.y.max(right.y);
    let x2 = left.x.saturating_add(left.width).min(right.x.saturating_add(right.width));
    let y2 = left.y.saturating_add(left.height).min(right.y.saturating_add(right.height));

    (x2 > x1 && y2 > y1).then(|| Rect::new(x1, y1, x2 - x1, y2 - y1))
}

/// Detect the same OK button in any bounded area, regardless of whether
/// the result popup replaced the enchant dialog. The wider scale range helps
/// when Windows display scaling differs between computers.
fn find_ok_button(
    frame: &Mat,
    search_rect: Rect,
    cfg: &EnchantDetectorConfig,
) -> opencv::Result<MatchResult> {
    let search = Mat::roi(frame, search_rect)?.try_clone()?;
    let source = grayscale(&search)?;
    let template = grayscale(&decode_image(OK_BUTTON_TEMPLATE)?)?;
    let match_cfg = EnchantDetectorConfig {
        scale_min: OK_BUTTON_SCALE_MIN,
        scale_max: OK_BUTTON_SCALE_MAX,
        scale_step: OK_BUTTON_SCALE_STEP,
        ..cfg.clone()
    };

    let mut matched = match_template(&source, &template, &match_cfg, cfg.confidence_threshold)?;
    if let Some(rect) = &mut matched.rect {
        rect.x += search_rect.x;
        rect.y += search_rect.y;
    }
    Ok(matched)
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
    // The supplied game capture has a dark but saturated blue button (V≈53),
    // so a high value floor misses the active state. The gray button has low S.
    let lower = Scalar::new(92.0, 85.0, 25.0, 0.0);
    let upper = Scalar::new(140.0, 255.0, 255.0, 0.0);
    let mut mask = Mat::default();
    core::in_range(&hsv, &lower, &upper, &mut mask)?;
    Ok(mask)
}

fn find_enchant_dialog(
    frame: &Mat,
    cfg: &EnchantDetectorConfig,
) -> opencv::Result<Option<Rect>> {
    let source = grayscale(frame)?;
    let header = grayscale(&decode_image(ENCHANT_HEADER_TEMPLATE)?)?;
    let matched = match_template(&source, &header, cfg, cfg.header_threshold.clamp(0.35, 0.99))?;
    if !matched.found {
        return Ok(None);
    }

    let Some(header_rect) = matched.rect else {
        return Ok(None);
    };

    // Reference capture geometry: the 299x23 header begins 3 px to the
    // right of the 302x200 enchant dialog's top-left corner.
    let scale = matched.scale;
    let dialog = Region {
        x: header_rect.x - (3.0 * scale).round() as i32,
        y: header_rect.y,
        width: (302.0 * scale).round() as i32,
        height: (200.0 * scale).round() as i32,
    };
    Ok(clip_region(dialog, frame))
}

fn empty_match() -> MatchResult {
    MatchResult {
        found: false,
        score: 0.0,
        rect: None,
        scale: 1.0,
    }
}

fn detect_start_button(
    frame: &Mat,
    dialog_rect: Rect,
    cfg: &EnchantDetectorConfig,
) -> opencv::Result<Option<StartButtonState>> {
    let rect = if cfg.start_button_region.is_configured() {
        let Some(rect) = clip_region(cfg.start_button_region, frame) else {
            return Ok(None);
        };
        if rect.width != cfg.start_button_region.width
            || rect.height != cfg.start_button_region.height
            || rect.x < dialog_rect.x
            || rect.y < dialog_rect.y
            || rect.x + rect.width > dialog_rect.x + dialog_rect.width
            || rect.y + rect.height > dialog_rect.y + dialog_rect.height
        {
            return Ok(None);
        }
        rect
    } else {
        // The supplied Start crop is matched only inside the detected enchant
        // dialog, so the similar Buy/Close buttons cannot be selected.
        let dialog_crop = Mat::roi(frame, dialog_rect)?.try_clone()?;
        let source = grayscale(&dialog_crop)?;
        let template = grayscale(&decode_image(START_TEMPLATE)?)?;
        let matched = match_template(
            &source,
            &template,
            cfg,
            cfg.start_template_threshold,
        )?;
        if !matched.found {
            return Ok(None);
        }
        let Some(local_rect) = matched.rect else {
            return Ok(None);
        };
        Rect::new(
            dialog_rect.x + local_rect.x,
            dialog_rect.y + local_rect.y,
            local_rect.width,
            local_rect.height,
        )
    };

    let crop = Mat::roi(frame, rect)?.try_clone()?;
    let mask = blue_mask(&crop)?;
    let ratio = f64::from(core::count_non_zero(&mask)?)
        / f64::from((rect.width * rect.height).max(1));

    Ok(Some(StartButtonState {
        active: ratio >= cfg.min_blue_ratio,
        rect,
    }))
}

pub fn analyze_enchant_window(
    frame: &Mat,
    cfg: &EnchantDetectorConfig,
) -> opencv::Result<EnchantAnalysis> {
    analyze_enchant_window_inner(frame, cfg, false)
}

/// Result-screen mode searches the centered half-area of the captured game
/// window using the same OK-button template as the in-dialog detector.
pub fn analyze_enchant_result(
    frame: &Mat,
    cfg: &EnchantDetectorConfig,
) -> opencv::Result<EnchantAnalysis> {
    analyze_enchant_window_inner(frame, cfg, true)
}

fn analyze_enchant_window_inner(
    frame: &Mat,
    cfg: &EnchantDetectorConfig,
    search_result_popup: bool,
) -> opencv::Result<EnchantAnalysis> {
    if frame.empty() {
        return Err(opencv::Error::new(core::StsBadArg, "Входной кадр пуст"));
    }

    let dialog_window = if search_result_popup {
        None
    } else {
        find_enchant_dialog(frame, cfg)?
    };
    let mut ok_button = empty_match();
    let mut start_button = None;

    if !search_result_popup {
        if let Some(dialog) = dialog_window {
            let search_rect = if let Some(region) = cfg.search_region {
                let requested = clip_region(region, frame).ok_or_else(|| {
                    opencv::Error::new(core::StsBadArg, "Область поиска ОК вне кадра")
                })?;
                let x1 = requested.x.max(dialog.x);
                let y1 = requested.y.max(dialog.y);
                let x2 = (requested.x + requested.width).min(dialog.x + dialog.width);
                let y2 = (requested.y + requested.height).min(dialog.y + dialog.height);
                if x2 <= x1 || y2 <= y1 {
                    return Err(opencv::Error::new(
                        core::StsBadArg,
                        "Область поиска ОК не пересекается с окном зачарования",
                    ));
                }
                Rect::new(x1, y1, x2 - x1, y2 - y1)
            } else {
                dialog
            };

            ok_button = find_ok_button(frame, search_rect, cfg)?;

            start_button = detect_start_button(frame, dialog, cfg)?;
        }
    }

    if search_result_popup {
        // Result popups may appear anywhere in the captured game window.
        // Ignore dialog-only ROIs so this path always searches the full frame.
        let full_frame = Rect::new(0, 0, frame.cols(), frame.rows());
        ok_button = find_ok_button(frame, full_frame, cfg)?;
    }

    Ok(EnchantAnalysis {
        dialog_window,
        ok_button,
        start_button,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_sized_frame() -> Mat {
        let mut frame = Mat::new_rows_cols_with_default(
            222,
            329,
            core::CV_8UC3,
            Scalar::new(28.0, 35.0, 42.0, 0.0),
        )
        .unwrap();

        let header = decode_image(ENCHANT_HEADER_TEMPLATE).unwrap();
        let mut header_roi = Mat::roi_mut(&mut frame, Rect::new(25, 7, 299, 23)).unwrap();
        header.copy_to(&mut header_roi).unwrap();

        let start = decode_image(START_TEMPLATE).unwrap();
        let mut start_roi = Mat::roi_mut(&mut frame, Rect::new(53, 158, 71, 28)).unwrap();
        start.copy_to(&mut start_roi).unwrap();

        frame
    }

    #[test]
    fn ok_button_is_found_in_full_frame_search() {
        let mut frame = Mat::new_rows_cols_with_default(
            480,
            640,
            core::CV_8UC3,
            Scalar::new(28.0, 35.0, 42.0, 0.0),
        )
        .unwrap();

        let ok = decode_image(OK_BUTTON_TEMPLATE).unwrap();
        let expected_rect = Rect::new(
            (frame.cols() - ok.cols()) / 2 + 24,
            (frame.rows() - ok.rows()) / 2 - 18,
            ok.cols(),
            ok.rows(),
        );
        let mut ok_roi = Mat::roi_mut(&mut frame, expected_rect).unwrap();
        ok.copy_to(&mut ok_roi).unwrap();

        let cfg = EnchantDetectorConfig {
            confidence_threshold: 0.95,
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = analyze_enchant_result(&frame, &cfg).unwrap();
        assert!(result.ok_button.found, "score={}", result.ok_button.score);
        assert!(result.ok_button.score >= cfg.confidence_threshold);
        assert_eq!(result.ok_button.rect, Some(expected_rect));
    }

    #[test]
    fn ok_button_is_found_with_135_percent_display_scaling() {
        let mut frame = Mat::new_rows_cols_with_default(
            480,
            640,
            core::CV_8UC3,
            Scalar::new(28.0, 35.0, 42.0, 0.0),
        )
        .unwrap();
        let template = grayscale(&decode_image(OK_BUTTON_TEMPLATE).unwrap()).unwrap();
        let mut scaled = Mat::default();
        imgproc::resize(
            &template,
            &mut scaled,
            Size::new(
                (f64::from(template.cols()) * 1.35).round() as i32,
                (f64::from(template.rows()) * 1.35).round() as i32,
            ),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )
        .unwrap();
        let mut scaled_bgr = Mat::default();
        imgproc::cvt_color(
            &scaled,
            &mut scaled_bgr,
            imgproc::COLOR_GRAY2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap();

        let expected_rect = Rect::new(
            (frame.cols() - scaled.cols()) / 2 + 32,
            (frame.rows() - scaled.rows()) / 2 - 20,
            scaled.cols(),
            scaled.rows(),
        );
        let mut roi = Mat::roi_mut(&mut frame, expected_rect).unwrap();
        scaled_bgr.copy_to(&mut roi).unwrap();

        let result = analyze_enchant_result(&frame, &EnchantDetectorConfig::default()).unwrap();
        assert!(result.ok_button.found, "score={}", result.ok_button.score);
        assert_eq!(result.ok_button.rect, Some(expected_rect));
    }

    #[test]
    fn ok_button_at_frame_edge_is_detected() {
        let mut frame = Mat::new_rows_cols_with_default(
            260,
            420,
            core::CV_8UC3,
            Scalar::new(28.0, 35.0, 42.0, 0.0),
        )
        .unwrap();
        let ok = decode_image(OK_BUTTON_TEMPLATE).unwrap();
        let mut roi = Mat::roi_mut(&mut frame, Rect::new(8, 8, ok.cols(), ok.rows())).unwrap();
        ok.copy_to(&mut roi).unwrap();

        let cfg = EnchantDetectorConfig {
            confidence_threshold: 0.95,
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = analyze_enchant_result(&frame, &cfg).unwrap();
        assert!(result.ok_button.found, "score={}", result.ok_button.score);
        assert_eq!(result.ok_button.rect, Some(Rect::new(8, 8, ok.cols(), ok.rows())));
    }

    #[test]
    fn same_ok_button_is_detected_when_enchant_dialog_has_closed() {
        let mut frame = Mat::new_rows_cols_with_default(
            480,
            640,
            core::CV_8UC3,
            Scalar::new(28.0, 35.0, 42.0, 0.0),
        )
        .unwrap();
        let ok = decode_image(OK_BUTTON_TEMPLATE).unwrap();
        let expected_rect = Rect::new(
            (frame.cols() - ok.cols()) / 2,
            (frame.rows() - ok.rows()) / 2,
            ok.cols(),
            ok.rows(),
        );
        let mut roi = Mat::roi_mut(&mut frame, expected_rect).unwrap();
        ok.copy_to(&mut roi).unwrap();

        let cfg = EnchantDetectorConfig {
            confidence_threshold: 0.95,
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = analyze_enchant_result(&frame, &cfg).unwrap();
        assert!(result.dialog_window.is_none());
        assert!(result.ok_button.found, "score={}", result.ok_button.score);
        assert_eq!(result.ok_button.rect, Some(expected_rect));
    }

    #[test]
    fn single_ok_button_template_matches_itself() {
        let frame = grayscale(&decode_image(OK_BUTTON_TEMPLATE).unwrap()).unwrap();
        let cfg = EnchantDetectorConfig {
            confidence_threshold: 0.95,
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = match_template(&frame, &frame, &cfg, cfg.confidence_threshold).unwrap();
        assert!(result.found, "score={}", result.score);
        assert!(result.score > 0.99);
    }

    #[test]
    fn single_ok_template_matches_a_scaled_button() {
        let template = grayscale(&decode_image(OK_BUTTON_TEMPLATE).unwrap()).unwrap();
        let mut scaled = Mat::default();
        imgproc::resize(
            &template,
            &mut scaled,
            Size::new(
                (f64::from(template.cols()) * 1.15).round() as i32,
                (f64::from(template.rows()) * 1.15).round() as i32,
            ),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )
        .unwrap();

        let cfg = EnchantDetectorConfig {
            scale_min: 0.80,
            scale_max: 1.25,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = match_template(&scaled, &template, &cfg, cfg.confidence_threshold).unwrap();
        assert!(result.found, "scaled button score={}", result.score);
    }

    #[test]
    fn active_start_template_is_found_only_inside_detected_enchant_dialog() {
        let frame = reference_sized_frame();
        let cfg = EnchantDetectorConfig {
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = analyze_enchant_window(&frame, &cfg).unwrap();
        assert_eq!(result.dialog_window, Some(Rect::new(22, 7, 302, 200)));
        let start = result.start_button.expect("Start template should be detected");
        assert!(start.active);
        assert_eq!((start.rect.x, start.rect.y), (53, 158));
    }

    #[test]
    fn same_ok_button_template_is_used_inside_enchant_dialog() {
        let mut frame = reference_sized_frame();
        let ok = decode_image(OK_BUTTON_TEMPLATE).unwrap();
        let expected = Rect::new(124, 88, ok.cols(), ok.rows());
        let mut roi = Mat::roi_mut(&mut frame, expected).unwrap();
        ok.copy_to(&mut roi).unwrap();

        let cfg = EnchantDetectorConfig {
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = analyze_enchant_window(&frame, &cfg).unwrap();
        assert!(result.ok_button.found, "score={}", result.ok_button.score);
        assert_eq!(result.ok_button.rect, Some(expected));
    }

    #[test]
    fn gray_start_button_is_detected_but_not_active() {
        let mut frame = reference_sized_frame();
        let start = grayscale(&decode_image(START_TEMPLATE).unwrap()).unwrap();
        let mut gray = Mat::default();
        imgproc::cvt_color(
            &start,
            &mut gray,
            imgproc::COLOR_GRAY2BGR,
            0,
            core::AlgorithmHint::ALGO_HINT_DEFAULT,
        )
        .unwrap();
        let mut roi = Mat::roi_mut(&mut frame, Rect::new(53, 158, 71, 28)).unwrap();
        gray.copy_to(&mut roi).unwrap();

        let cfg = EnchantDetectorConfig {
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = analyze_enchant_window(&frame, &cfg).unwrap();
        let start = result.start_button.expect("Gray Start outline should match");
        assert!(!start.active);
    }

    #[test]
    fn dialog_header_is_required_before_start_or_ok_detection() {
        let frame = Mat::new_rows_cols_with_default(
            222,
            329,
            core::CV_8UC3,
            Scalar::new(80.0, 80.0, 80.0, 0.0),
        )
        .unwrap();
        let result = analyze_enchant_window(&frame, &EnchantDetectorConfig::default()).unwrap();
        assert!(result.dialog_window.is_none());
        assert!(result.start_button.is_none());
        assert!(!result.ok_button.found);
    }

    #[test]
    fn configured_start_roi_detects_active_button() {
        let frame = reference_sized_frame();
        let cfg = EnchantDetectorConfig {
            start_button_region: Region {
                x: 53,
                y: 158,
                width: 71,
                height: 28,
            },
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = analyze_enchant_window(&frame, &cfg).unwrap();
        let button = result.start_button.unwrap();
        assert!(button.active);
    }
}
