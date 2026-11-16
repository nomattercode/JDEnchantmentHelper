use opencv::{
    core::{self, Mat, Rect, Scalar},
    imgproc,
    prelude::*,
};

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
    pub threshold: f64,
}

impl Default for InventoryConfig {
    fn default() -> Self {
        Self {
            // Estimate measured from the supplied 317x478 screenshot.
            origin_x: 34, origin_y: 84, columns: 6, rows: 8,
            cell_width: 36, cell_height: 36, step_x: 40, step_y: 40,
            inset: 6, start_row: 0, start_col: 1, amulet_row: 0, amulet_col: 0,
            threshold: 0.11,
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

fn score_cell(frame: &Mat, rect: Rect, config: &InventoryConfig) -> opencv::Result<f64> {
    let inset = config.inset.max(0).min((rect.width.min(rect.height) - 2).max(0) / 2);
    let roi = Rect::new(
        rect.x + inset,
        rect.y + inset,
        rect.width - 2 * inset,
        rect.height - 2 * inset,
    );
    let crop = Mat::roi(frame, roi)?;
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
                let cell = CellDebug { row, col, rect, score: 0.0, kind: CellKind::OutOfFrame };
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

            let cell = CellDebug { row, col, rect, score, kind };
            cells.push(cell);
        }
    }

    Ok(InventoryScan { cells })
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
        let amulet = scan.cells.iter().find(|x| x.row == 0 && x.col == 0);
        assert!(amulet.is_some_and(|cell| cell.kind == CellKind::Amulet));
        assert!(!scan.cells.iter().any(|cell| {
            cell.row == 0 && cell.col == 0 && cell.kind == CellKind::Occupied
        }));
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
