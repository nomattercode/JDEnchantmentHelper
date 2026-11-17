use serde::{Deserialize, Serialize};
use std::{env, fs, io::Write, path::PathBuf};
use crate::{model::MAX_PASSES, vision::{EnchantDetectorConfig, InventoryConfig, Region}};

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
    fn default() -> Self { Self { schema_version: 1, passes_total: 1, inventory: InventoryConfig::default(), enchant: EnchantDetectorConfig::default(), inventory_grid_region: Region::default() } }
}
impl UserSettings {
    pub fn normalize(&mut self) {
        self.schema_version=1;
        self.passes_total=self.passes_total.clamp(1,MAX_PASSES);
        self.inventory.columns=self.inventory.columns.clamp(1,12); self.inventory.rows=self.inventory.rows.clamp(1,30);
        self.inventory.cell_width=self.inventory.cell_width.clamp(4,4096); self.inventory.cell_height=self.inventory.cell_height.clamp(4,4096);
        self.inventory.step_x=self.inventory.step_x.clamp(self.inventory.cell_width,4096); self.inventory.step_y=self.inventory.step_y.clamp(self.inventory.cell_height,4096);
        self.inventory.origin_x=self.inventory.origin_x.clamp(0,32768); self.inventory.origin_y=self.inventory.origin_y.clamp(0,32768);
        self.inventory.inset=self.inventory.inset.clamp(0,128);
        self.inventory.amulet_row=self.inventory.amulet_row.min(self.inventory.rows-1); self.inventory.amulet_col=self.inventory.amulet_col.min(self.inventory.columns-1);
        self.inventory.start_row=self.inventory.start_row.min(self.inventory.rows-1); self.inventory.start_col=self.inventory.start_col.min(self.inventory.columns-1);
        self.inventory.threshold=unit(self.inventory.threshold,0.11); self.inventory.header_threshold=unit(self.inventory.header_threshold,0.72);
        self.enchant.confidence_threshold=unit(self.enchant.confidence_threshold,0.78); self.enchant.header_threshold=unit(self.enchant.header_threshold,0.64);
        self.enchant.start_template_threshold=unit(self.enchant.start_template_threshold,0.68); self.enchant.min_blue_ratio=unit(self.enchant.min_blue_ratio,0.45);
        if !self.enchant.scale_min.is_finite(){self.enchant.scale_min=0.65;} if !self.enchant.scale_max.is_finite(){self.enchant.scale_max=1.45;} if !self.enchant.scale_step.is_finite(){self.enchant.scale_step=0.05;}
        self.enchant.scale_min=self.enchant.scale_min.clamp(0.25,2.5); self.enchant.scale_max=self.enchant.scale_max.clamp(self.enchant.scale_min,3.0); self.enchant.scale_step=self.enchant.scale_step.clamp(0.02,0.25);
        if !self.inventory_grid_region.is_configured()||self.inventory_grid_region.x<0||self.inventory_grid_region.y<0 {self.inventory_grid_region=Region::default();}
    }
}
fn unit(v:f64,fallback:f64)->f64{if v.is_finite(){v.clamp(0.0,1.0)}else{fallback}}
pub fn settings_path()->Result<PathBuf,String>{
    let root=env::var_os("APPDATA").map(PathBuf::from).or_else(||env::var_os("XDG_CONFIG_HOME").map(PathBuf::from)).or_else(||env::var_os("HOME").map(|h|PathBuf::from(h).join(".config"))).ok_or_else(||"Не найдена системная папка настроек (APPDATA).".to_string())?;
    Ok(root.join("JDEnchantmentHelper").join("settings.json"))
}
pub fn load()->Result<UserSettings,String>{
    let path=settings_path()?; if !path.exists(){return Ok(UserSettings::default());}
    let data=fs::read_to_string(&path).map_err(|e|format!("Не удалось прочитать {}: {e}",path.display()))?;
    let mut value:UserSettings=serde_json::from_str(&data).map_err(|e|format!("Файл настроек {} повреждён: {e}",path.display()))?;
    value.normalize();Ok(value)
}
pub fn save(value:&UserSettings)->Result<(),String>{
    let path=settings_path()?;let parent=path.parent().ok_or_else(||"У файла настроек нет родительской папки.".to_string())?;
    fs::create_dir_all(parent).map_err(|e|format!("Не удалось создать {}: {e}",parent.display()))?;
    let mut normalized=value.clone();normalized.normalize();let data=serde_json::to_vec_pretty(&normalized).map_err(|e|format!("Ошибка сериализации настроек: {e}"))?;
    let mut temp=tempfile::NamedTempFile::new_in(parent).map_err(|e|format!("Не удалось создать временный файл настроек: {e}"))?;
    temp.write_all(&data).map_err(|e|format!("Не удалось записать настройки: {e}"))?;temp.as_file().sync_all().map_err(|e|format!("Не удалось синхронизировать настройки: {e}"))?;
    temp.persist(&path).map_err(|e|format!("Не удалось заменить {}: {}",path.display(),e.error))?;Ok(())
}
#[cfg(test)]mod tests{use super::*;#[test]fn normalize_clamps_invalid_sensitivity_and_geometry(){let mut s=UserSettings::default();s.inventory.threshold=f64::NAN;s.enchant.confidence_threshold=3.0;s.inventory.columns=0;s.passes_total=0;s.normalize();assert_eq!(s.inventory.threshold,0.11);assert_eq!(s.enchant.confidence_threshold,1.0);assert_eq!(s.inventory.columns,1);assert_eq!(s.passes_total,1);}}
