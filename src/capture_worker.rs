use opencv::{core, prelude::*};
use std::{
    sync::mpsc::{self, Receiver, SyncSender, TryRecvError, TrySendError},
    thread,
};

use crate::{
    platform::capture_window,
    vision::{analyze_enchant_result, analyze_enchant_window, EnchantDetectorConfig},
};

#[derive(Clone, Copy, Debug)]
pub enum AnalysisKind { None, EnchantDialog, ResultPopup }

struct CaptureRequest {
    id: u64,
    handle: isize,
    analysis: AnalysisKind,
    config: EnchantDetectorConfig,
}

pub struct CaptureResponse {
    pub id: u64,
    pub result: Result<CapturedFrame, String>,
}

pub struct CapturedFrame {
    pub width: i32,
    pub height: i32,
    pub pixels_bgr: Vec<u8>,
    pub analysis: Option<Result<crate::vision::enchant_detector::EnchantAnalysis, String>>,
}

impl CapturedFrame {
    pub fn to_mat(&self) -> Result<core::Mat, String> {
        if self.width <= 0 || self.height <= 0 {
            return Err("Получен кадр с некорректным размером.".into());
        }
        let expected = usize::try_from(self.width)
            .ok()
            .and_then(|w| usize::try_from(self.height).ok().and_then(|h| w.checked_mul(h)))
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or_else(|| "Размер кадра выходит за допустимый диапазон.".to_string())?;
        if self.pixels_bgr.len() != expected {
            return Err(format!(
                "Повреждён буфер кадра: ожидалось {expected} байт, получено {}.",
                self.pixels_bgr.len()
            ));
        }
        let mut frame = core::Mat::new_rows_cols_with_default(
            self.height, self.width, core::CV_8UC3, core::Scalar::all(0.0),
        ).map_err(|error| format!("Не удалось восстановить кадр: {error}"))?;
        frame.data_bytes_mut()
            .map_err(|error| format!("Не удалось получить буфер кадра: {error}"))?
            .copy_from_slice(&self.pixels_bgr);
        Ok(frame)
    }
}

pub struct CaptureWorker {
    requests: SyncSender<CaptureRequest>,
    responses: Receiver<CaptureResponse>,
    next_id: u64,
}

impl CaptureWorker {
    pub fn new() -> Result<Self, String> {
        let (request_tx, request_rx) = mpsc::sync_channel::<CaptureRequest>(1);
        let (response_tx, response_rx) = mpsc::channel::<CaptureResponse>();
        thread::Builder::new()
            .name("jade-dynasty-capture".into())
            .spawn(move || {
                while let Ok(request) = request_rx.recv() {
                    let result = (|| {
                        let frame = capture_window(request.handle)?;
                        let analysis = match request.analysis {
                            AnalysisKind::None => None,
                            AnalysisKind::EnchantDialog => Some(
                                analyze_enchant_window(&frame, &request.config)
                                    .map_err(|error| error.to_string()),
                            ),
                            AnalysisKind::ResultPopup => Some(
                                analyze_enchant_result(&frame, &request.config)
                                    .map_err(|error| error.to_string()),
                            ),
                        };
                        let continuous = frame.try_clone()
                            .map_err(|error| format!("Не удалось подготовить кадр для потока: {error}"))?;
                        let pixels_bgr = continuous.data_bytes()
                            .map_err(|error| format!("Не удалось прочитать пиксели кадра: {error}"))?
                            .to_vec();
                        Ok(CapturedFrame {
                            width: continuous.cols(),
                            height: continuous.rows(),
                            pixels_bgr,
                            analysis,
                        })
                    })();
                    if response_tx.send(CaptureResponse { id: request.id, result }).is_err() {
                        break;
                    }
                }
            })
            .map_err(|error| format!("Не удалось запустить поток захвата кадров: {error}"))?;
        Ok(Self { requests: request_tx, responses: response_rx, next_id: 0 })
    }

    pub fn request(&mut self, handle: isize, analysis: AnalysisKind, config: EnchantDetectorConfig) -> Result<u64, String> {
        let id = self.next_id.wrapping_add(1);
        self.requests.try_send(CaptureRequest { id, handle, analysis, config })
            .map_err(|error| match error {
                TrySendError::Full(_) => "Поток захвата ещё обрабатывает предыдущий кадр.".to_string(),
                TrySendError::Disconnected(_) => "Поток захвата завершился.".to_string(),
            })?;
        self.next_id = id;
        Ok(id)
    }

    pub fn try_receive(&self) -> Result<Option<CaptureResponse>, String> {
        match self.responses.try_recv() {
            Ok(response) => Ok(Some(response)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err("Поток захвата завершился.".into()),
        }
    }
}
