#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RunState {
    #[default]
    Idle,
    Running,
    Paused,
    Stopped,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GameWindow {
    /// Native window handle, stored as an integer so the UI model is
    /// independent from Windows-specific pointer types.
    pub handle: isize,
    pub title: String,
    pub process_id: u32,
    pub width: u32,
    pub height: u32,
}

pub const MAX_PASSES: u8 = 10;

#[cfg(test)]
mod tests {
    use super::{RunState, MAX_PASSES};

    #[test]
    fn pass_limit_is_ten() {
        assert_eq!(MAX_PASSES, 10);
    }
}
