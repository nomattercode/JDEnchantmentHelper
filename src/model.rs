#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RunState {
    #[default]
    Idle,
    Running,
    Paused,
    Stopped,
}

impl RunState {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Idle => "Ожидание",
            Self::Running => "Запущена",
            Self::Paused => "Пауза",
            Self::Stopped => "Остановлена",
        }
    }
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

pub const MAX_RUNS: u8 = 10;

#[cfg(test)]
mod tests {
    use super::{RunState, MAX_RUNS};

    #[test]
    fn run_limit_is_ten() {
        assert_eq!(MAX_RUNS, 10);
    }

    #[test]
    fn states_have_user_facing_labels() {
        assert_eq!(RunState::Idle.label(), "Ожидание");
        assert_eq!(RunState::Running.label(), "Запущена");
        assert_eq!(RunState::Paused.label(), "Пауза");
        assert_eq!(RunState::Stopped.label(), "Остановлена");
    }
}
