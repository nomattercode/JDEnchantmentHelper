use opencv::{
    core::{self, Mat, Point, Rect, Scalar, Size},
    imgproc,
    prelude::*,
};

use crate::vision::image_io::decode_image;

const SUCCESS_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/ok_success.png");
const FAILURE_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/ok_failure.png");
const START_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/start_button_active.png");
const ENCHANT_HEADER_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/enchant_header.png");

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
    pub start_template_threshold: f64,
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
    pub dialog_window: Option<Rect>,
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
        found: best_score >= threshold,
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
    let matched = match_template(&source, &header, "enchant_header", cfg, 0.64)?;
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

fn empty_match(name: &str) -> MatchResult {
    MatchResult {
        found: false,
        score: 0.0,
        rect: None,
        template_name: name.into(),
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
            "start_button",
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
        blue_ratio: ratio,
        rect,
    }))
}

pub fn analyze_enchant_window(
    frame: &Mat,
    cfg: &EnchantDetectorConfig,
) -> opencv::Result<EnchantAnalysis> {
    analyze_enchant_window_inner(frame, cfg, false)
}

/// Result-screen mode searches the whole captured game window for OK.
/// It is used only after Start was clicked, when the game may replace or cover
/// the enchant dialog with a separate result popup.
pub fn analyze_enchant_result(
    frame: &Mat,
    cfg: &EnchantDetectorConfig,
) -> opencv::Result<EnchantAnalysis> {
    analyze_enchant_window_inner(frame, cfg, true)
}

fn analyze_enchant_window_inner(
    frame: &Mat,
    cfg: &EnchantDetectorConfig,
    search_ok_full_frame: bool,
) -> opencv::Result<EnchantAnalysis> {
    if frame.empty() {
        return Err(opencv::Error::new(core::StsBadArg, "Входной кадр пуст"));
    }

    let dialog_window = find_enchant_dialog(frame, cfg)?;
    let mut ok_button = empty_match("ok");
    let mut start_button = None;
    let mut overlay = frame.try_clone()?;
    let mut log = vec![format!(
        "[enchant] frame={}x{} dialog_threshold=0.64 OK threshold={:.3} START threshold={:.3}",
        frame.cols(),
        frame.rows(),
        cfg.confidence_threshold,
        cfg.start_template_threshold
    )];

    if let Some(dialog) = dialog_window.clone() {
        imgproc::rectangle(
            &mut overlay,
            dialog,
            Scalar::new(255.0, 190.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            0,
        )?;
        log.push(format!(
            "[enchant] dialog detected rect=({}, {}, {}, {})",
            dialog.x, dialog.y, dialog.width, dialog.height
        ));

        let search_rect = if let Some(region) = cfg.search_region {
            let requested = clip_region(region, frame).ok_or_else(|| {
                opencv::Error::new(core::StsBadArg, "Область поиска OK вне кадра")
            })?;
            let x1 = requested.x.max(dialog.x);
            let y1 = requested.y.max(dialog.y);
            let x2 = (requested.x + requested.width).min(dialog.x + dialog.width);
            let y2 = (requested.y + requested.height).min(dialog.y + dialog.height);
            if x2 <= x1 || y2 <= y1 {
                return Err(opencv::Error::new(
                    core::StsBadArg,
                    "Область поиска OK не пересекается с окном зачарования",
                ));
            }
            Rect::new(x1, y1, x2 - x1, y2 - y1)
        } else {
            dialog
        };

        let search = Mat::roi(frame, search_rect)?.try_clone()?;
        let source = grayscale(&search)?;
        let success = grayscale(&decode_image(SUCCESS_TEMPLATE)?)?;
        let failure = grayscale(&decode_image(FAILURE_TEMPLATE)?)?;

        let mut best = match_template(
            &source,
            &success,
            "ok_success",
            cfg,
            cfg.confidence_threshold,
        )?;
        let failure_match = match_template(
            &source,
            &failure,
            "ok_failure",
            cfg,
            cfg.confidence_threshold,
        )?;
        if failure_match.score > best.score {
            best = failure_match;
        }
        if let Some(rect) = &mut best.rect {
            rect.x += search_rect.x;
            rect.y += search_rect.y;
        }
        ok_button = best;

        if let Some(rect) = ok_button.rect {
            let color = if ok_button.found {
                Scalar::new(0.0, 220.0, 0.0, 0.0)
            } else {
                Scalar::new(0.0, 180.0, 255.0, 0.0)
            };
            imgproc::rectangle(&mut overlay, rect, color, 2, imgproc::LINE_8, 0)?;
            imgproc::put_text(
                &mut overlay,
                &format!("{} {:.3}", ok_button.template_name, ok_button.score),
                Point::new(rect.x, (rect.y - 5).max(14)),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }
        log.push(format!(
            "[enchant] OK template={} score={:.4} threshold={:.3} found={}",
            ok_button.template_name,
            ok_button.score,
            cfg.confidence_threshold,
            ok_button.found
        ));

        start_button = detect_start_button(frame, dialog, cfg)?;
    } else {
        log.push("[enchant] enchant dialog header not found; Start detection is disabled".into());
    }

    if search_ok_full_frame {
        let search_rect = if let Some(region) = cfg.search_region {
            clip_region(region, frame).ok_or_else(|| {
                opencv::Error::new(core::StsBadArg, "Область поиска OK вне кадра")
            })?
        } else {
            Rect::new(0, 0, frame.cols(), frame.rows())
        };
        let search = Mat::roi(frame, search_rect)?.try_clone()?;
        let source = grayscale(&search)?;
        let success = grayscale(&decode_image(SUCCESS_TEMPLATE)?)?;
        let failure = grayscale(&decode_image(FAILURE_TEMPLATE)?)?;

        let mut best = match_template(
            &source,
            &success,
            "ok_success",
            cfg,
            cfg.confidence_threshold,
        )?;
        let failure_match = match_template(
            &source,
            &failure,
            "ok_failure",
            cfg,
            cfg.confidence_threshold,
        )?;
        if failure_match.score > best.score {
            best = failure_match;
        }
        if let Some(rect) = &mut best.rect {
            rect.x += search_rect.x;
            rect.y += search_rect.y;
        }

        // Prefer the strongest whole-frame match. This covers both cases:
        // an OK button inside the enchant dialog and a separate result popup
        // elsewhere on the game surface.
        if best.score > ok_button.score {
            ok_button = best;
        }

        log.push(format!(
            "[enchant] full-frame result search: template={} score={:.4} threshold={:.3} found={} rect={:?}",
            ok_button.template_name,
            ok_button.score,
            cfg.confidence_threshold,
            ok_button.found,
            ok_button.rect
        ));

        if let Some(rect) = ok_button.rect {
            let color = if ok_button.found {
                Scalar::new(0.0, 220.0, 0.0, 0.0)
            } else {
                Scalar::new(0.0, 180.0, 255.0, 0.0)
            };
            imgproc::rectangle(&mut overlay, rect, color, 2, imgproc::LINE_8, 0)?;
            imgproc::put_text(
                &mut overlay,
                &format!("{} {:.3}", ok_button.template_name, ok_button.score),
                Point::new(rect.x, (rect.y - 5).max(14)),
                imgproc::FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                imgproc::LINE_AA,
                false,
            )?;
        }
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
            "[enchant] START score template threshold={:.3} blue_ratio={:.3} threshold={:.3} active={} rect=({}, {}, {}, {})",
            cfg.start_template_threshold,
            button.blue_ratio,
            cfg.min_blue_ratio,
            button.active,
            button.rect.x,
            button.rect.y,
            button.rect.width,
            button.rect.height
        ));
    } else if dialog_window.is_some() {
        log.push("[enchant] Start button template not found in the enchant dialog".into());
    }

    Ok(EnchantAnalysis {
        dialog_window,
        ok_button,
        start_button,
        overlay,
        log,
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
    fn result_ok_is_found_outside_enchant_dialog() {
        let mut frame = Mat::new_rows_cols_with_default(
            340,
            500,
            core::CV_8UC3,
            Scalar::new(28.0, 35.0, 42.0, 0.0),
        )
        .unwrap();

        let header = decode_image(ENCHANT_HEADER_TEMPLATE).unwrap();
        let mut header_roi = Mat::roi_mut(&mut frame, Rect::new(25, 7, 299, 23)).unwrap();
        header.copy_to(&mut header_roi).unwrap();

        let ok = decode_image(SUCCESS_TEMPLATE).unwrap();
        let mut ok_roi = Mat::roi_mut(&mut frame, Rect::new(350, 240, ok.cols(), ok.rows())).unwrap();
        ok.copy_to(&mut ok_roi).unwrap();

        let cfg = EnchantDetectorConfig {
            confidence_threshold: 0.95,
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = analyze_enchant_result(&frame, &cfg).unwrap();
        assert!(result.dialog_window.is_some());
        assert!(result.ok_button.found, "score={}", result.ok_button.score);
        assert_eq!(
            result.ok_button.rect,
            Some(Rect::new(350, 240, ok.cols(), ok.rows()))
        );
    }

    #[test]
    fn result_ok_is_detected_when_enchant_dialog_has_closed() {
        let mut frame = Mat::new_rows_cols_with_default(
            260,
            420,
            core::CV_8UC3,
            Scalar::new(28.0, 35.0, 42.0, 0.0),
        )
        .unwrap();
        let ok = decode_image(FAILURE_TEMPLATE).unwrap();
        let mut roi = Mat::roi_mut(&mut frame, Rect::new(240, 160, ok.cols(), ok.rows())).unwrap();
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
        assert_eq!(
            result.ok_button.rect,
            Some(Rect::new(240, 160, ok.cols(), ok.rows()))
        );
    }

    #[test]
    fn success_ok_template_matches_itself() {
        let frame = grayscale(&decode_image(SUCCESS_TEMPLATE).unwrap()).unwrap();
        let cfg = EnchantDetectorConfig {
            confidence_threshold: 0.95,
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = match_template(&frame, &frame, "ok_success", &cfg, 0.95).unwrap();
        assert!(result.found, "score={}", result.score);
        assert!(result.score > 0.99);
    }

    #[test]
    fn failure_ok_template_matches_itself() {
        let frame = grayscale(&decode_image(FAILURE_TEMPLATE).unwrap()).unwrap();
        let cfg = EnchantDetectorConfig {
            confidence_threshold: 0.95,
            scale_min: 1.0,
            scale_max: 1.0,
            scale_step: 0.05,
            ..Default::default()
        };
        let result = match_template(&frame, &frame, "ok_failure", &cfg, 0.95).unwrap();
        assert!(result.found, "score={}", result.score);
        assert!(result.score > 0.99);
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
        assert!(start.active, "blue_ratio={}", start.blue_ratio);
        assert!(start.blue_ratio >= cfg.min_blue_ratio);
        assert_eq!((start.rect.x, start.rect.y), (53, 158));
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
        assert!(start.blue_ratio < cfg.min_blue_ratio);
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
    fn configured_start_roi_reports_blue_ratio() {
        let mut frame = reference_sized_frame();
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
        assert!(button.blue_ratio > 0.90);
    }
}
