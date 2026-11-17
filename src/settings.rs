use serde::{Deserialize, Serialize};
use std::{env, fs, io::Write, path::PathBuf};

use crate::{
    model::MAX_PASSES,
    vision::{EnchantDetectorConfig, InventoryConfig, Region},
};

const SETTINGS_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct UserSettings {
    pub schema_version: u32,
    pub passes_total: u8,
    pub inventory: InventoryConfig,
    pub enchant: EnchantDetectorConfig,
    pub inventory_grid_region: Region,
}

impl Default for UserSettings {
    fn default() -> Self {
        Self {
            schema_version: SETTINGS_SCHEMA_VERSION,
            passes_total: 1,
            inventory: InventoryConfig::default(),
            enchant: EnchantDetectorConfig::default(),
            inventory_grid_region: Region::default(),
        }
    }
}

impl UserSettings {
    pub fn normalize(&mut self) {
        self.schema_version = SETTINGS_SCHEMA_VERSION;
        self.passes_total = self.passes_total.clamp(1, MAX_PASSES);

        self.inventory.columns = self.inventory.columns.clamp(1, 12);
        self.inventory.rows = self.inventory.rows.clamp(1, 30);
        self.inventory.cell_width = self.inventory.cell_width.clamp(4, 4096);
        self.inventory.cell_height = self.inventory.cell_height.clamp(4, 4096);
        self.inventory.step_x = self.inventory.step_x.clamp(self.inventory.cell_width, 4096);
        self.inventory.step_y = self.inventory.step_y.clamp(self.inventory.cell_height, 4096);
        self.inventory.origin_x = self.inventory.origin_x.clamp(0, 32_768);
        self.inventory.origin_y = self.inventory.origin_y.clamp(0, 32_768);
        self.inventory.inset = self.inventory.inset.clamp(0, 128);
        self.inventory.amulet_row = self.inventory.amulet_row.min(self.inventory.rows - 1);
        self.inventory.amulet_col = self.inventory.amulet_col.min(self.inventory.columns - 1);
        self.inventory.start_row = self.inventory.start_row.min(self.inventory.rows - 1);
        self.inventory.start_col = self.inventory.start_col.min(self.inventory.columns - 1);
        self.inventory.threshold = unit_interval(self.inventory.threshold, 0.11);
        self.inventory.header_threshold = unit_interval(self.inventory.header_threshold, 0.72);

        self.enchant.confidence_threshold = unit_interval(self.enchant.confidence_threshold, 0.78);
        self.enchant.header_threshold = unit_interval(self.enchant.header_threshold, 0.64);
        self.enchant.start_template_threshold =
            unit_interval(self.enchant.start_template_threshold, 0.68);
        self.enchant.min_blue_ratio = unit_interval(self.enchant.min_blue_ratio, 0.45);

        if !self.enchant.scale_min.is_finite() {
            self.enchant.scale_min = 0.65;
        }
        if !self.enchant.scale_max.is_finite() {
            self.enchant.scale_max = 1.45;
        }
        if !self.enchant.scale_step.is_finite() {
            self.enchant.scale_step = 0.05;
        }
        self.enchant.scale_min = self.enchant.scale_min.clamp(0.25, 2.5);
        self.enchant.scale_max = self.enchant.scale_max.clamp(self.enchant.scale_min, 3.0);
        self.enchant.scale_step = self.enchant.scale_step.clamp(0.02, 0.25);

        let region = self.inventory_grid_region;
        if !region.is_configured()
            || region.x < 0
            || region.y < 0
            || region.width > 32_768
            || region.height > 32_768
        {
            self.inventory_grid_region = Region::default();
        }
    }
}

fn unit_interval(value: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        fallback
    }
}

pub fn settings_path() -> Result<PathBuf, String> {
    let root = env::var_os("APPDATA")
        .map(PathBuf::from)
        .or_else(|| env::var_os("XDG_CONFIG_HOME").map(PathBuf::from))
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))
        .ok_or_else(|| "Не найдена системная папка настроек (APPDATA).".to_string())?;

    Ok(root.join("JDEnchantmentHelper").join("settings.json"))
}

pub fn load() -> Result<UserSettings, String> {
    let path = settings_path()?;
    if !path.exists() {
        return Ok(UserSettings::default());
    }

    let contents = fs::read_to_string(&path)
        .map_err(|error| format!("Не удалось прочитать {}: {error}", path.display()))?;
    let mut settings: UserSettings = serde_json::from_str(&contents)
        .map_err(|error| format!("Файл настроек {} повреждён: {error}", path.display()))?;
    settings.normalize();
    Ok(settings)
}

pub fn save(settings: &UserSettings) -> Result<(), String> {
    let path = settings_path()?;
    let parent = path
        .parent()
        .ok_or_else(|| "У файла настроек нет родительской папки.".to_string())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("Не удалось создать {}: {error}", parent.display()))?;

    let mut normalized = settings.clone();
    normalized.normalize();
    let contents = serde_json::to_vec_pretty(&normalized)
        .map_err(|error| format!("Не удалось сериализовать настройки: {error}"))?;

    // Place the temporary file beside the destination so replacement stays
    // on the same volume and can be atomic on Windows.
    let mut temporary = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| format!("Не удалось создать временный файл настроек: {error}"))?;
    temporary
        .write_all(&contents)
        .map_err(|error| format!("Не удалось записать настройки: {error}"))?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|error| format!("Не удалось синхронизировать настройки: {error}"))?;
    temporary.persist(&path).map_err(|error| {
        format!(
            "Не удалось атомарно заменить файл настроек {}: {}",
            path.display(),
            error.error
        )
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_clamps_invalid_sensitivity_and_geometry() {
        let mut settings = UserSettings::default();
        settings.inventory.threshold = f64::NAN;
        settings.enchant.confidence_threshold = 3.0;
        settings.inventory.columns = 0;
        settings.passes_total = 0;
        settings.normalize();

        assert_eq!(settings.inventory.threshold, 0.11);
        assert_eq!(settings.enchant.confidence_threshold, 1.0);
        assert_eq!(settings.inventory.columns, 1);
        assert_eq!(settings.passes_total, 1);
    }

    #[test]
    fn settings_round_trip_preserves_sensitivity_values() {
        let mut expected = UserSettings::default();
        expected.inventory.threshold = 0.23;
        expected.inventory.header_threshold = 0.81;
        expected.enchant.confidence_threshold = 0.93;

        let encoded = serde_json::to_string(&expected).unwrap();
        let actual: UserSettings = serde_json::from_str(&encoded).unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn older_settings_without_new_threshold_fields_use_defaults() {
        let encoded = r#"{
            "schema_version": 1,
            "passes_total": 2,
            "inventory": { "columns": 6, "rows": 8 },
            "enchant": {}
        }"#;
        let mut settings: UserSettings = serde_json::from_str(encoded).unwrap();
        settings.normalize();

        assert_eq!(settings.passes_total, 2);
        assert_eq!(settings.inventory.header_threshold, 0.72);
        assert_eq!(settings.enchant.header_threshold, 0.64);
    }
}
