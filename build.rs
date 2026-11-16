use embed_manifest::{embed_manifest, manifest::ExecutionLevel, new_manifest};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_REQUIRE_ADMIN");

    let is_windows = std::env::var_os("CARGO_CFG_WINDOWS").is_some();
    let require_admin = std::env::var_os("CARGO_FEATURE_REQUIRE_ADMIN").is_some();

    // Do not put requireAdministrator on Cargo's test harness executable:
    // Cargo launches tests without elevation and Windows would reject it (OS error 740).
    if is_windows && require_admin {
        let manifest = new_manifest("JD Enchantment Helper")
            .requested_execution_level(ExecutionLevel::RequireAdministrator);
        embed_manifest(manifest)
            .expect("Не удалось встроить манифест запроса прав администратора");
    }
}
