use opencv::{
    core::{self, Mat, Point, Rect, Scalar},
    imgproc,
    prelude::*,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectionMethod {
    GrayscaleVariance,
    EdgeDensity,
    Saturation,
    Combined,
}

impl DetectionMethod {
    pub const ALL: [Self; 4] = [
        Self::GrayscaleVariance, Self::EdgeDensity, Self::Saturation, Self::Combined,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::GrayscaleVariance => "Контраст (среднеквадратичное отклонение)",
            Self::EdgeDensity => "Плотность границ",
            Self::Saturation => "Насыщенность цвета",
            Self::Combined => "Комбинированный",
        }
    }
}

#[derive(Clone, Debug)]
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
    pub method: DetectionMethod,
    pub threshold: f64,
    pub canny_low: f64,
    pub canny_high: f64,
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            // Estimate measured from the supplied 317x478 screenshot.
            origin_x: 34, origin_y: 84, columns: 6, rows: 8,
            cell_width: 36, cell_height: 36, step_x: 40, step_y: 40,
            inset: 6, start_row: 0, start_col: 1, amulet_row: 0, amulet_col: 0,
            method: DetectionMethod::GrayscaleVariance, threshold: 0.11,
            canny_low: 45.0, canny_high: 130.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellKind { Amulet, BeforeStart, Occupied, Empty, OutOfFrame }

impl CellKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Amulet => "АМУЛЕТ",
            Self::BeforeStart => "ПРОПУСК",
            Self::Occupied => "ПРЕДМЕТ",
            Self::Empty => "ПУСТО",
            Self::OutOfFrame => "ВНЕ КАДРА",
        }
    }

    fn color(self) -> Scalar {
        match self {
            Self::Amulet => Scalar::new(0.0, 165.0, 255.0, 0.0),
            Self::BeforeStart => Scalar::new(110.0, 110.0, 110.0, 0.0),
            Self::Occupied => Scalar::new(40.0, 220.0, 40.0, 0.0),
            Self::Empty => Scalar::new(40.0, 80.0, 240.0, 0.0),
            Self::OutOfFrame => Scalar::new(180.0, 180.0, 180.0, 0.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CellDebug {
    pub row: u8,
    pub col: u8,
    pub rect: Rect,
    pub score: f64,
    pub kind: CellKind,
}

#[derive(Debug)]
pub struct InventoryScan {
    pub cells: Vec<CellDebug>,
    /// Occupied cells at/after the start point, excluding the amulet cell.
    pub candidates: Vec<CellDebug>,
    pub overlay: Mat,
    pub score_map: Mat,
    pub log: Vec<String>,
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

fn score_cell(frame: &Mat, rect: Rect, config: &InventoryConfig) -> opencv::Result<f64> {
    let inset = config.inset.max(0).min((rect.width.min(rect.height) - 2).max(0) / 2);
    let roi = Rect::new(rect.x + inset, rect.y + inset, rect.width - 2 * inset, rect.height - 2 * inset);
    let crop = Mat::roi(frame, roi)?;
    let mut gray = Mat::default();
    imgproc::cvt_color(&crop, &mut gray, imgproc::COLOR_BGR2GRAY, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;

    let mut mean = Scalar::default();
    let mut stddev = Scalar::default();
    core::mean_std_dev(&gray, &mut mean, &mut stddev, &core::no_array())?;
    let variance_score = (stddev[0] / 64.0).clamp(0.0, 1.0);

    let mut edges = Mat::default();
    imgproc::canny(&gray, &mut edges, config.canny_low, config.canny_high, 3, false)?;
    let edge_score = (
        f64::from(core::count_non_zero(&edges)?) /
        f64::from(roi.width * roi.height) / 0.12
    ).clamp(0.0, 1.0);

    let mut hsv = Mat::default();
    imgproc::cvt_color(&crop, &mut hsv, imgproc::COLOR_BGR2HSV, 0, core::AlgorithmHint::ALGO_HINT_DEFAULT)?;
    let mut channels = opencv::core::Vector::<Mat>::new();
    core::split(&hsv, &mut channels)?;
    let saturation = channels.get(1)?;
    let saturation_score = core::mean(&saturation, &core::no_array())?[0] / 255.0;

    Ok(match config.method {
        DetectionMethod::GrayscaleVariance => variance_score,
        DetectionMethod::EdgeDensity => edge_score,
        DetectionMethod::Saturation => saturation_score,
        DetectionMethod::Combined => 0.55 * variance_score + 0.30 * edge_score + 0.15 * saturation_score,
    })
}

fn draw_cell(image: &mut Mat, cell: &CellDebug) -> opencv::Result<()> {
    imgproc::rectangle(image, cell.rect, cell.kind.color(), 1, imgproc::LINE_8, 0)?;
    let label = format!("{} {:.2}", cell.kind.label(), cell.score);
    imgproc::put_text(
        image, &label,
        Point::new(cell.rect.x, (cell.rect.y + 10).min(image.rows().saturating_sub(2))),
        imgproc::FONT_HERSHEY_PLAIN, 0.65, cell.kind.color(), 1, imgproc::LINE_AA, false,
    )?;
    Ok(())
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

    let mut overlay = frame.try_clone()?;
    let mut score_map = Mat::zeros(frame.rows(), frame.cols(), core::CV_8UC1)?.to_mat()?;
    let mut cells = Vec::with_capacity(usize::from(config.columns) * usize::from(config.rows));
    let mut candidates = Vec::new();
    let mut log = vec![format!(
        "[inventory] frame={}x{} grid={}x{} origin=({}, {}) cell={}x{} step={}x{} method={:?} threshold={:.3}",
        frame.cols(), frame.rows(), config.columns, config.rows, config.origin_x, config.origin_y,
        config.cell_width, config.cell_height, config.step_x, config.step_y, config.method, config.threshold
    )];

    for row in 0..config.rows {
        for col in 0..config.columns {
            let rect = cell_rect(config, row, col);
            if !inside_frame(rect, frame) {
                let cell = CellDebug { row, col, rect, score: 0.0, kind: CellKind::OutOfFrame };
                log.push(format!("[inventory] r{}c{} outside frame", row + 1, col + 1));
                draw_cell(&mut overlay, &cell)?;
                cells.push(cell);
                continue;
            }

            let score = score_cell(frame, rect, config)?;
            // The amulet is excluded first, even if the start coordinate is earlier.
            let is_amulet = row == config.amulet_row && col == config.amulet_col;
            let before_start = row < config.start_row || (row == config.start_row && col < config.start_col);
            let kind = if is_amulet { CellKind::Amulet }
                else if before_start { CellKind::BeforeStart }
                else if score >= config.threshold { CellKind::Occupied }
                else { CellKind::Empty };

            let intensity = (score * 255.0).round().clamp(0.0, 255.0);
            imgproc::rectangle(&mut score_map, rect, Scalar::all(intensity), imgproc::FILLED, imgproc::LINE_8, 0)?;
            let cell = CellDebug { row, col, rect, score, kind };
            log.push(format!(
                "[inventory] r{}c{} score={:.3} threshold={:.3} => {} rect=({}, {}, {}, {})",
                row + 1, col + 1, score, config.threshold, kind.label(),
                rect.x, rect.y, rect.width, rect.height
            ));
            if kind == CellKind::Occupied { candidates.push(cell.clone()); }
            draw_cell(&mut overlay, &cell)?;
            cells.push(cell);
        }
    }

    log.push(format!(
        "[inventory] candidates={} amulet=r{}c{} start=r{}c{}",
        candidates.len(), config.amulet_row + 1, config.amulet_col + 1,
        config.start_row + 1, config.start_col + 1
    ));
    Ok(InventoryScan { cells, candidates, overlay, score_map, log })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(scan.cells.iter().any(|x| x.kind == CellKind::Amulet));
        assert!(!scan.candidates.iter().any(|x| x.row == 0 && x.col == 0));
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
        assert_eq!(scan.candidates.len(), 2);
    }
}
