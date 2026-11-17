use serde::{Deserialize, Serialize};
use opencv::{
    core::{self, Mat, Rect, Scalar, Size},
    imgproc,
    prelude::*,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct InventoryConfig {
    pub origin_x: i32,
    pub origin_y: i32,
    pub columns: u8,
    pub rows: u8,
    pub cell_width: i32,
    pub cell_height: i32,
    pub step_x: i32,
    pub step_y: i32,
    pub inset: i32,
    pub start_row: u8,
    pub start_col: u8,
    pub amulet_row: u8,
    pub amulet_col: u8,
    pub threshold: f64,
    pub header_threshold: f64,
    /// OpenCV HSV hue tolerance around red (0/179 wraparound).
    pub red_hue_range: u8,
    pub red_saturation_min: u8,
    pub red_value_min: u8,
    /// Minimum fraction of red pixels in a slot's inner crop.
    pub red_ratio_threshold: f64,
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            // Estimate measured from the supplied 317x478 screenshot.
            origin_x: 34, origin_y: 84, columns: 6, rows: 8,
            cell_width: 36, cell_height: 36, step_x: 40, step_y: 40,
            inset: 6, start_row: 0, start_col: 1, amulet_row: 0, amulet_col: 0,
            threshold: 0.11,
            header_threshold: 0.72,
            red_hue_range: 12,
            red_saturation_min: 80,
            red_value_min: 35,
            red_ratio_threshold: 0.02,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellKind { Amulet, BeforeStart, Occupied, Empty, OutOfFrame }

#[derive(Clone, Debug)]
pub struct CellDebug {
    pub row: u8,
    pub col: u8,
    pub rect: Rect,
    pub score: f64,
    pub red_ratio: f64,
    pub enchantable: bool,
    pub kind: CellKind,
}

#[derive(Debug)]
pub struct InventoryScan {
    pub cells: Vec<CellDebug>,
}

fn cell_rect(config: &InventoryConfig, row: u8, col: u8) -> Rect {
    Rect::new(
        config.origin_x + i32::from(col) * config.step_x,
        config.origin_y + i32::from(row) * config.step_y,
        config.cell_width,
        config.cell_height,
    )
}

fn inside_frame(rect: Rect, frame: &Mat) -> bool {
    rect.x >= 0 && rect.y >= 0 && rect.width > 0 && rect.height > 0
        && rect.x + rect.width <= frame.cols()
        && rect.y + rect.height <= frame.rows()
}

fn inner_cell_rect(rect: Rect, config: &InventoryConfig) -> Rect {
    let inset = config.inset.max(0).min((rect.width.min(rect.height) - 2).max(0) / 2);
    Rect::new(
        rect.x + inset,
        rect.y + inset,
        rect.width - 2 * inset,
        rect.height - 2 * inset,
    )
}

fn score_cell(frame: &Mat, rect: Rect, config: &InventoryConfig) -> opencv::Result<f64> {
    let crop = Mat::roi(frame, inner_cell_rect(rect, config))?;
    let mut gray = Mat::default();
    imgproc::cvt_color(
        &crop,
        &mut gray,
        imgproc::COLOR_BGR2GRAY,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    let mut mean = Scalar::default();
    let mut stddev = Scalar::default();
    core::mean_std_dev(&gray, &mut mean, &mut stddev, &core::no_array())?;
    Ok((stddev[0] / 64.0).clamp(0.0, 1.0))
}

fn red_ratio_cell(frame: &Mat, rect: Rect, config: &InventoryConfig) -> opencv::Result<f64> {
    let crop_rect = inner_cell_rect(rect, config);
    let crop = Mat::roi(frame, crop_rect)?;
    let mut hsv = Mat::default();
    imgproc::cvt_color(
        &crop,
        &mut hsv,
        imgproc::COLOR_BGR2HSV,
        0,
        core::AlgorithmHint::ALGO_HINT_DEFAULT,
    )?;

    // Red wraps around the ends of OpenCV's 0..179 hue range.
    let hue_range = config.red_hue_range.clamp(1, 30);
    let saturation = f64::from(config.red_saturation_min);
    let value = f64::from(config.red_value_min);
    let mut low_red = Mat::default();
    let mut high_red = Mat::default();
    core::in_range(
        &hsv,
        &Scalar::new(0.0, saturation, value, 0.0),
        &Scalar::new(f64::from(hue_range), 255.0, 255.0, 0.0),
        &mut low_red,
    )?;
    core::in_range(
        &hsv,
        &Scalar::new(f64::from(180_u8.saturating_sub(hue_range)), saturation, value, 0.0),
        &Scalar::new(179.0, 255.0, 255.0, 0.0),
        &mut high_red,
    )?;

    let mut red_mask = Mat::default();
    core::bitwise_or(&low_red, &high_red, &mut red_mask, &core::no_array())?;
    let red_pixels = f64::from(core::count_non_zero(&red_mask)?);
    let total_pixels = f64::from(crop_rect.width) * f64::from(crop_rect.height);
    Ok(if total_pixels > 0.0 { red_pixels / total_pixels } else { 0.0 })
}

pub fn scan_inventory(frame: &Mat, config: &InventoryConfig) -> opencv::Result<InventoryScan> {
    if frame.empty() {
        return Err(opencv::Error::new(core::StsBadArg, "Входной кадр пуст"));
    }
    if config.columns == 0 || config.rows == 0 || config.cell_width < 4 || config.cell_height < 4 {
        return Err(opencv::Error::new(core::StsBadArg, "Некорректная геометрия сетки"));
    }
    if config.step_x < config.cell_width || config.step_y < config.cell_height {
        return Err(opencv::Error::new(core::StsBadArg, "Шаг сетки не может быть меньше размера ячейки"));
    }

    let mut cells = Vec::with_capacity(usize::from(config.columns) * usize::from(config.rows));
    for row in 0..config.rows {
        for col in 0..config.columns {
            let rect = cell_rect(config, row, col);
            if !inside_frame(rect, frame) {
                let cell = CellDebug {
                    row,
                    col,
                    rect,
                    score: 0.0,
                    red_ratio: 0.0,
                    enchantable: false,
                    kind: CellKind::OutOfFrame,
                };
                cells.push(cell);
                continue;
            }

            let score = score_cell(frame, rect, config)?;
            let red_ratio = red_ratio_cell(frame, rect, config)?;
            // The amulet is excluded first, even if the start coordinate is earlier.
            let is_amulet = row == config.amulet_row && col == config.amulet_col;
            let before_start = row < config.start_row || (row == config.start_row && col < config.start_col);
            let kind = if is_amulet { CellKind::Amulet }
                else if before_start { CellKind::BeforeStart }
                else if score >= config.threshold { CellKind::Occupied }
                else { CellKind::Empty };
            let enchantable = !is_amulet
                && !before_start
                && red_ratio >= config.red_ratio_threshold;

            let cell = CellDebug { row, col, rect, score, red_ratio, enchantable, kind };
            cells.push(cell);
        }
    }

    Ok(InventoryScan { cells })
}


const INVENTORY_HEADER_TEMPLATE: &[u8] = include_bytes!("../../assets/templates/inventory_header.png");

#[derive(Clone, Debug)]
pub struct InventoryDetection {
    pub header_rect: Rect,
    pub grid_region: crate::vision::enchant_detector::Region,
    pub config: InventoryConfig,
    pub score: f64,
}

/// Match the inventory title panel and infer the grid offsets from the
/// attached reference capture. The manual calibration remains a fallback.
pub fn detect_inventory_grid(frame: &Mat, base: &InventoryConfig) -> opencv::Result<Option<InventoryDetection>> {
    if frame.empty() { return Err(opencv::Error::new(core::StsBadArg, "Входной кадр пуст")); }
    let header = crate::vision::image_io::decode_image(INVENTORY_HEADER_TEMPLATE)?;
    let mut source = Mat::default();
    let mut template = Mat::default();
    imgproc::cvt_color(frame, &mut source, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    imgproc::cvt_color(&header, &mut template, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let (mut best_score, mut best_rect, mut best_scale) = (-1.0_f64, None, 1.0_f64);
    let mut scale = 0.70_f64;
    while scale <= 3.00 + 0.0001 {
        let width = (f64::from(template.cols()) * scale).round() as i32;
        let height = (f64::from(template.rows()) * scale).round() as i32;
        if width >= 30 && height >= 12 && width <= source.cols() && height <= source.rows() {
            let mut resized = Mat::default();
            imgproc::resize(&template, &mut resized, Size::new(width, height), 0.0, 0.0,
                if scale > 1.0 { imgproc::INTER_LINEAR } else { imgproc::INTER_AREA })?;
            let mut response = Mat::default();
            imgproc::match_template(&source, &resized, &mut response, imgproc::TM_CCOEFF_NORMED, &Mat::default())?;
            let mut max_value = 0.0;
            let mut max_point = core::Point::default();
            core::min_max_loc(&response, None, Some(&mut max_value), None, Some(&mut max_point), &core::no_array())?;
            if max_value > best_score {
                best_score = max_value;
                best_rect = Some(Rect::new(max_point.x, max_point.y, width, height));
                best_scale = scale;
            }
        }
        scale += 0.05;
    }
    if best_score < base.header_threshold.clamp(0.35, 0.99) { return Ok(None); }
    let Some(header_rect) = best_rect else { return Ok(None); };

    // Template crop is 87×30 (half-size); scale converts offsets into
    // client pixels: origin (+9,+31), pitch 20×20, cell 18×18.
    let mut config = base.clone();
    // The supplied inventory layout has 6 columns and 8 rows. Determine
    // geometry from the matched header while retaining user slot preferences.
    config.columns = 6;
    config.rows = 8;
    config.origin_x = header_rect.x + (9.0 * best_scale).round() as i32;
    config.origin_y = header_rect.y + (31.0 * best_scale).round() as i32;
    config.step_x = (20.0 * best_scale).round() as i32;
    config.step_y = (20.0 * best_scale).round() as i32;
    config.cell_width = (18.0 * best_scale).round() as i32;
    config.cell_height = (18.0 * best_scale).round() as i32;
    config.inset = (3.0 * best_scale).round() as i32;
    let grid_region = crate::vision::enchant_detector::Region {
        x: config.origin_x,
        y: config.origin_y,
        width: config.cell_width + config.step_x * i32::from(config.columns.saturating_sub(1)),
        height: config.cell_height + config.step_y * i32::from(config.rows.saturating_sub(1)),
    };
    if grid_region.x < 0 || grid_region.y < 0
        || grid_region.x.saturating_add(grid_region.width) > frame.cols()
        || grid_region.y.saturating_add(grid_region.height) > frame.rows() {
        return Ok(None);
    }
    Ok(Some(InventoryDetection { header_rect, grid_region, config, score: best_score }))
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn inventory_header_finds_grid_using_reference_offsets() {
        let mut frame = Mat::new_rows_cols_with_default(478, 317, core::CV_8UC3, Scalar::all(28.0)).unwrap();
        let header = crate::vision::image_io::decode_image(INVENTORY_HEADER_TEMPLATE).unwrap();
        let mut scaled = Mat::default();
        imgproc::resize(&header, &mut scaled, Size::new(174, 60), 0.0, 0.0, imgproc::INTER_LINEAR).unwrap();
        let mut roi = Mat::roi_mut(&mut frame, Rect::new(16, 22, 174, 60)).unwrap();
        scaled.copy_to(&mut roi).unwrap();
        let found = detect_inventory_grid(&frame, &InventoryConfig::default()).unwrap().unwrap();
        assert!(found.score >= 0.72);
        assert_eq!((found.grid_region.x, found.grid_region.y), (34, 84));
        assert_eq!((found.grid_region.width, found.grid_region.height), (236, 316));
        assert_eq!((found.config.columns, found.config.rows), (6, 8));
    }

    #[test]
    fn inventory_header_detection_supports_three_x_ui_scale() {
        let mut frame = Mat::new_rows_cols_with_default(
            620, 650, core::CV_8UC3, Scalar::all(28.0)
        ).unwrap();
        let header = crate::vision::image_io::decode_image(INVENTORY_HEADER_TEMPLATE).unwrap();
        let mut scaled = Mat::default();
        imgproc::resize(
            &header, &mut scaled, Size::new(261, 90), 0.0, 0.0, imgproc::INTER_LINEAR
        ).unwrap();
        let mut roi = Mat::roi_mut(&mut frame, Rect::new(30, 20, 261, 90)).unwrap();
        scaled.copy_to(&mut roi).unwrap();

        let found = detect_inventory_grid(&frame, &InventoryConfig::default()).unwrap().unwrap();
        assert!(found.score >= 0.72);
        assert!((found.grid_region.x - 57).abs() <= 2);
        assert!((found.grid_region.y - 113).abs() <= 2);
        assert!((found.grid_region.width - 354).abs() <= 2);
        assert!((found.grid_region.height - 474).abs() <= 2);
    }

    #[test]
    fn inventory_header_detector_rejects_a_blank_frame() {
        let frame = Mat::new_rows_cols_with_default(
            478,
            317,
            core::CV_8UC3,
            Scalar::all(28.0),
        )
        .unwrap();
        let found = detect_inventory_grid(&frame, &InventoryConfig::default()).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn default_grid_matches_supplied_inventory_reference() {
        let c = InventoryConfig::default();
        assert_eq!((c.origin_x, c.origin_y), (34, 84));
        assert_eq!((c.columns, c.rows), (6, 8));
        assert_eq!((c.cell_width, c.cell_height), (36, 36));
        assert_eq!((c.step_x, c.step_y), (40, 40));
    }

    #[test]
    fn amulet_is_excluded_even_when_start_is_before_it() {
        let mut c = InventoryConfig::default();
        c.start_row = 0; c.start_col = 0;
        let frame = Mat::new_rows_cols_with_default(430, 320, core::CV_8UC3, Scalar::all(0.0)).unwrap();
        let scan = scan_inventory(&frame, &c).unwrap();
        let amulet = scan.cells.iter().find(|x| x.row == 0 && x.col == 0);
        assert!(amulet.is_some_and(|cell| cell.kind == CellKind::Amulet));
        assert!(amulet.is_some_and(|cell| !cell.enchantable));
        assert!(!scan.cells.iter().any(|cell| {
            cell.row == 0 && cell.col == 0 && cell.kind == CellKind::Occupied
        }));
    }

    #[test]
    fn red_pixels_are_the_authoritative_enchantability_signal() {
        let mut config = InventoryConfig::default();
        config.columns = 1;
        config.rows = 1;
        config.start_row = 0;
        config.start_col = 0;
        config.amulet_row = u8::MAX;
        config.amulet_col = u8::MAX;
        config.threshold = 1.0;
        config.red_ratio_threshold = 0.02;

        let mut frame = Mat::new_rows_cols_with_default(
            64, 64, core::CV_8UC3, Scalar::all(45.0)
        ).unwrap();
        imgproc::rectangle(
            &mut frame,
            Rect::new(21, 21, 5, 5),
            Scalar::new(0.0, 0.0, 255.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        ).unwrap();

        let scan = scan_inventory(&frame, &config).unwrap();
        let cell = &scan.cells[0];
        // The small red area deliberately does not pass the old grayscale
        // occupancy score, proving that the red marker is an independent signal.
        assert_eq!(cell.kind, CellKind::Empty);
        assert!(cell.score < config.threshold);
        assert!(cell.red_ratio >= config.red_ratio_threshold);
        assert!(cell.enchantable);
    }

    #[test]
    fn high_end_of_hsv_hue_range_is_also_counted_as_red() {
        let mut config = InventoryConfig::default();
        config.columns = 1;
        config.rows = 1;
        config.start_row = 0;
        config.start_col = 0;
        config.amulet_row = u8::MAX;
        config.amulet_col = u8::MAX;
        config.threshold = 1.0;

        let mut frame = Mat::new_rows_cols_with_default(
            64, 64, core::CV_8UC3, Scalar::all(45.0)
        ).unwrap();
        // BGR (8, 0, 255) is a red hue close to OpenCV's upper hue boundary.
        imgproc::rectangle(
            &mut frame,
            Rect::new(21, 21, 5, 5),
            Scalar::new(8.0, 0.0, 255.0, 0.0),
            -1,
            imgproc::LINE_8,
            0,
        ).unwrap();

        let scan = scan_inventory(&frame, &config).unwrap();
        assert!(scan.cells[0].enchantable);
    }

    #[test]
    fn occupied_slot_without_red_pixels_is_not_enchantable() {
        let mut config = InventoryConfig::default();
        config.columns = 1;
        config.rows = 1;
        config.start_row = 0;
        config.start_col = 0;
        config.amulet_row = u8::MAX;
        config.amulet_col = u8::MAX;
        config.threshold = 0.0;

        let frame = Mat::new_rows_cols_with_default(
            64, 64, core::CV_8UC3, Scalar::all(45.0)
        ).unwrap();
        let scan = scan_inventory(&frame, &config).unwrap();
        let cell = &scan.cells[0];
        assert_eq!(cell.kind, CellKind::Occupied);
        assert_eq!(cell.red_ratio, 0.0);
        assert!(!cell.enchantable);
    }

    #[test]
    fn start_cell_is_included_and_cells_before_it_are_skipped() {
        let mut c = InventoryConfig::default();
        c.rows = 1; c.columns = 3; c.start_row = 0; c.start_col = 1;
        c.amulet_col = 0; c.threshold = 0.0;
        let frame = Mat::new_rows_cols_with_default(140, 180, core::CV_8UC3, Scalar::all(20.0)).unwrap();
        let scan = scan_inventory(&frame, &c).unwrap();
        assert_eq!(scan.cells[0].kind, CellKind::Amulet);
        assert_eq!(scan.cells[1].kind, CellKind::Occupied);
        assert_eq!(scan.cells[2].kind, CellKind::Occupied);
        assert_eq!(
            scan.cells.iter().filter(|cell| cell.kind == CellKind::Occupied).count(),
            2
        );
    }
}
