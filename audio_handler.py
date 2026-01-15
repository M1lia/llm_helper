import logging
import queue
import threading
import wave
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional
import asyncio
import numpy as np
import sounddevice as sd
from typing import Callable
# Настройка логирования
logger = logging.getLogger(__name__)


@dataclass
class AudioMetrics:
    """Метрики для отслеживания работы аудио захвата"""
    frames_captured: int = 0
    frames_dropped: int = 0
    bytes_processed: int = 0

    def __str__(self) -> str:
        drop_rate = (self.frames_dropped / max(1, self.frames_captured + self.frames_dropped)) * 100
        return (f"Захвачено: {self.frames_captured}, Пропущено: {self.frames_dropped} "
                f"({drop_rate:.2f}%), Обработано: {self.bytes_processed / 1024:.1f} KB")


@dataclass
class AudioConfig:
    """Конфигурация аудио параметров"""
    sample_rate: int = 16000 
    channels: int = 1  
    dtype: str = 'int16' 
    blocksize: int = 2048  
    device: Optional[int] = None 
    queue_maxsize: int = 100  
    process_timeout: float = 0.5  


class AudioCapture:
    """Класс для захвата аудио с микрофона с асинхронной передачей"""

    def __init__(self, config: AudioConfig, data_handler: 'AudioDataHandler'):
        """
        Args:
            config: Конфигурация аудио
            data_handler: Объект для обработки/отправки данных
        """
        self.config = config
        self.data_handler = data_handler
        self.stream: Optional[sd.InputStream] = None
        self.audio_queue: queue.Queue = queue.Queue(maxsize=config.queue_maxsize)
        self._running: bool = False
        self._lock = threading.Lock()
        self._process_thread: Optional[threading.Thread] = None
        self.metrics = AudioMetrics()

    def _audio_callback(self, indata: np.ndarray, frames: int,
                        time: sd.CallbackTimeInfo, status: sd.CallbackFlags) -> None:
        """
        Callback функция, которая вызывается PortAudio при получении данных с микрофона
        Работает в отдельном потоке!

        Args:
            indata: Массив с аудио данными (shape: [frames, channels])
            frames: Количество фреймов в indata
            time: Временная информация
            status: Статус потока (ошибки, переполнения буфера и т.д.)
        """
        if status:
            logger.warning(f"Статус аудио потока: {status}")

        audio_chunk = indata.copy()

        try:
            # Неблокирующая попытка добавить в очередь
            self.audio_queue.put_nowait(audio_chunk)
            self.metrics.frames_captured += 1
        except queue.Full:
            self.metrics.frames_dropped += 1
            logger.warning("Очередь переполнена, фрейм пропущен")
        except Exception as e:
            logger.error(f"Ошибка в audio callback: {e}", exc_info=True)

    def start(self) -> None:
        """Запустить захват аудио"""
        if self._running:
            logger.warning("Захват уже запущен")
            return

        with self._lock:
            self._running = True

        try:
            # Создаем поток ввода (input stream)
            self.stream = sd.InputStream(
                device=self.config.device,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.blocksize,
                callback=self._audio_callback,
                latency='low'  # Минимальная задержка
            )

            self.stream.start()
            device_info = sd.query_devices(self.config.device)
            logger.info("Захват аудио запущен")
            logger.info(f"Устройство: {device_info['name']}")
            logger.info(f"Частота: {self.config.sample_rate} Гц, Каналы: {self.config.channels}, Бит: 16")

            # Запускаем фоновый поток для обработки очереди
            self._process_thread = threading.Thread(
                target=self._process_audio_queue,
                daemon=True,
                name="AudioProcessThread"
            )
            self._process_thread.start()

        except Exception as e:
            with self._lock:
                self._running = False
            logger.error(f"Ошибка при запуске захвата: {e}", exc_info=True)
            raise

    def _process_audio_queue(self) -> None:
        """Фоновый поток для обработки очереди аудио данных"""
        logger.debug("Поток обработки очереди запущен")

        while self._running:
            try:
                # Ждем аудио данные (timeout для проверки _running)
                audio_chunk = self.audio_queue.get(timeout=self.config.process_timeout)

                # Передаем в обработчик
                self.data_handler.handle_audio_chunk(audio_chunk)
                self.metrics.bytes_processed += audio_chunk.nbytes

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ошибка в обработке очереди: {e}", exc_info=True)

        logger.debug("Поток обработки очереди завершен")

    def stop(self) -> None:
        """Остановить захват аудио (graceful shutdown)"""
        logger.info("Остановка захвата аудио...")

        with self._lock:
            self._running = False

        # Ждем завершения обработки оставшихся данных в очереди
        if self._process_thread and self._process_thread.is_alive():
            logger.debug("Ожидание завершения потока обработки...")
            self._process_thread.join(timeout=2.0)
            if self._process_thread.is_alive():
                logger.warning("Поток обработки не завершился за 2 секунды")

        # Останавливаем stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        logger.info(f"Захват аудио остановлен. {self.metrics}")

    def get_metrics(self) -> AudioMetrics:
        """Получить текущие метрики"""
        return self.metrics

    # Context manager support
    def __enter__(self) -> 'AudioCapture':
        """Вход в context manager"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Выход из context manager"""
        self.stop()


class AudioDataHandler:
    """Базовый класс для обработки аудио данных"""
    
    def handle_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Обработать полученный аудио блок
        
        Args:
            audio_chunk: Массив аудио данных shape [frames, channels]
        """
        raise NotImplementedError


class AudioSender(AudioDataHandler):
    """Класс для отправки аудио данных с буферизацией"""

    def __init__(self,
                send_function: Callable[[bytes], None], 
                buffer_duration: float = 0.5,
                sample_rate: int = 24000, channels: int = 1,
                max_workers: int = 2
            ):
        """
        Args:
            buffer_duration: Сколько секунд аудио буферизировать перед отправкой
            sample_rate: Частота дискретизации
            channels: Количество каналов
            max_workers: Максимальное количество потоков для отправки
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = int(sample_rate * buffer_duration * channels)
        self.send_function = send_function

        # Используем deque для эффективного добавления элементов
        self.audio_buffer: deque = deque(maxlen=self.buffer_size * 10)

        self._lock = threading.Lock()
        self._send_count = 0

        # ThreadPoolExecutor для управления потоками отправки
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="AudioSender"
        )

    def handle_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """Буферизировать и отправить когда буфер полный"""
        chunk_flat = audio_chunk.flatten()

        with self._lock:
            self.audio_buffer.extend(chunk_flat)

            if len(self.audio_buffer) >= self.buffer_size:
                to_send = np.array([self.audio_buffer.popleft()
                                   for _ in range(self.buffer_size)], dtype='int16')

                # Отправляем через thread pool
                self._executor.submit(self._send_audio, to_send)

    def _send_audio(self, audio_data: np.ndarray) -> None:
        """Имитация отправки аудио данных"""
        with self._lock:
            self._send_count += 1
            count = self._send_count

        duration = len(audio_data) / self.sample_rate
        max_amplitude = np.max(np.abs(audio_data)) if len(audio_data) > 0 else 0

        logger.info(f"Отправка #{count}: {len(audio_data)} семплов "
                    f"({duration:.2f} сек), макс амплитуда: {max_amplitude}")

        if isinstance(self.send_function, Callable):
            try:
                audio_bytes = audio_data.tobytes()
                self.send_function(audio_bytes)
            except Exception as e:
                logger.error(f"Ошибка отправки: {e}")

    def close(self) -> None:
        """Закрыть sender и дождаться завершения всех отправок"""
        logger.info("Завершение AudioSender...")
        self._executor.shutdown(wait=True)
        logger.info(f"AudioSender закрыт. Всего отправлено: {self._send_count}")

    def __enter__(self) -> 'AudioSender':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class AudioWriter(AudioDataHandler):
    """Класс для записи аудио в WAV файл"""

    def __init__(self, filename: str, sample_rate: int = 16000, channels: int = 1):
        """
        Args:
            filename: Путь к WAV файлу
            sample_rate: Частота дискретизации
            channels: Количество каналов
        """
        self.filename = filename
        self.sample_rate = sample_rate
        self.channels = channels
        self.wav_file: Optional[wave.Wave_write] = None
        self._lock = threading.Lock()
        self._frames_written = 0
        self._init_wav()

    def _init_wav(self) -> None:
        """Инициализация WAV файла"""
        try:
            self.wav_file = wave.open(self.filename, 'wb')
            self.wav_file.setnchannels(self.channels)
            self.wav_file.setsampwidth(2)  # 16 бит = 2 байта
            self.wav_file.setframerate(self.sample_rate)
            logger.info(f"Создан WAV файл: {self.filename}")
        except Exception as e:
            logger.error(f"Ошибка создания WAV файла: {e}", exc_info=True)
            raise

    def handle_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """Записать аудио блок в файл"""
        with self._lock:
            if self.wav_file:
                try:
                    # Конвертируем в байты (int16 -> bytes)
                    audio_bytes = audio_chunk.astype('int16').tobytes()
                    self.wav_file.writeframes(audio_bytes)
                    self._frames_written += len(audio_chunk)
                except Exception as e:
                    logger.error(f"Ошибка записи в WAV файл: {e}", exc_info=True)

    def close(self) -> None:
        """Закрыть файл"""
        with self._lock:
            if self.wav_file:
                try:
                    self.wav_file.close()
                    duration = self._frames_written / self.sample_rate
                    logger.info(f"WAV файл закрыт: {self.filename} "
                               f"(записано {self._frames_written} фреймов, {duration:.2f} сек)")
                except Exception as e:
                    logger.error(f"Ошибка закрытия WAV файла: {e}", exc_info=True)
                finally:
                    self.wav_file = None

    def __enter__(self) -> 'AudioWriter':
        """Вход в context manager"""
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Выход из context manager"""
        self.close()


class CombinedHandler(AudioDataHandler):
    """Комбинированный обработчик для одновременно"""

    def __init__(self, handlers: List[AudioDataHandler]):
        """
        Args:
            handlers: Список обработчиков для последовательного вызова
        """
        self.handlers = handlers

    def handle_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """Передать chunk всем обработчикам"""
        for handler in self.handlers:
            try:
                handler.handle_audio_chunk(audio_chunk)
            except Exception as e:
                logger.error(f"Ошибка в обработчике {handler.__class__.__name__}: {e}", exc_info=True)


def main():
    """Главная функция для демонстрации работы"""
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Конфигурация
    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        dtype='int16',
        blocksize=2048,
        queue_maxsize=100,
        process_timeout=0.5,
        device=3  # Soundcore Life Tune
    )

    # Используем context managers для автоматического управления ресурсами
    try:
        with AudioSender(send_function = None, buffer_duration=0.5, sample_rate=config.sample_rate) as sender, \
             AudioWriter(filename="recording.wav", sample_rate=config.sample_rate) as writer, \
             AudioCapture(config, CombinedHandler([sender, writer])) as capture:

            logger.info("\nЗахватываю аудио (10 сек)...\n")
            sd.sleep(10000)

            metrics = capture.get_metrics()
            logger.info(f"\nМетрики: {metrics}")

    except KeyboardInterrupt:
        logger.info("\nПрервано пользователем")
    except Exception as e:
        logger.error(f"Ошибка: {e}", exc_info=True)
    finally:
        logger.info("\nГотово!")


if __name__ == "__main__":
    main()


#TODO implement