use embed_manifest::{embed_manifest, manifest::ExecutionLevel, new_manifest};

fn main() {
    if std::env::var_os("CARGO_CFG_WINDOWS").is_some() {
        let manifest = new_manifest("JD Enchantment Helper")
            .requested_execution_level(ExecutionLevel::RequireAdministrator);
        embed_manifest(manifest).expect("Не удалось встроить манифест запроса прав администратора");
    }

    println!("cargo:rerun-if-changed=build.rs");
}
