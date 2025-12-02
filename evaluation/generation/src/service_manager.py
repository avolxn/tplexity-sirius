"""
Модуль для автоматического управления сервисами, необходимыми для eval pipeline.
"""

import logging
import os
import subprocess
import time
import requests
from pathlib import Path
from typing import Optional, List, Dict
import atexit
import signal
import sys

logger = logging.getLogger(__name__)


class ServiceManager:
    """
    Менеджер для управления сервисами (generation, retriever).
    """
    
    def __init__(
        self,
        project_root: Optional[Path] = None,
        use_docker: bool = True,
        auto_stop: bool = True
    ):
        """
        Инициализация менеджера сервисов.
        
        Args:
            project_root: Корневая директория проекта (если None - определяется автоматически)
            use_docker: Использовать docker-compose для запуска (True) или запускать напрямую (False)
            auto_stop: Автоматически останавливать сервисы при завершении
        """
        if project_root is None:
            # Определяем корень проекта (3 уровня выше от eval/src/service_manager.py)
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.use_docker = use_docker
        self.auto_stop = auto_stop
        self.docker_compose_path = self.project_root / "docker-compose.yml"
        self.started_services: List[str] = []
        self.processes: Dict[str, subprocess.Popen] = {}
        
        # Регистрируем обработчик для остановки при завершении
        if auto_stop:
            atexit.register(self.stop_all_services)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для корректной остановки сервисов."""
        logger.info(f"Получен сигнал {signum}, останавливаем сервисы...")
        self.stop_all_services()
        sys.exit(0)
    
    def check_service_health(self, url: str, timeout: int = 5) -> bool:
        """
        Проверяет доступность сервиса по health check endpoint.
        
        Args:
            url: URL для проверки (например, http://localhost:8022/health)
            timeout: Таймаут в секундах
            
        Returns:
            True если сервис доступен, False иначе
        """
        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Сервис {url} недоступен: {e}")
            return False
    
    def wait_for_service(
        self,
        url: str,
        max_wait: int = 120,
        check_interval: int = 2
    ) -> bool:
        """
        Ожидает готовности сервиса.
        
        Args:
            url: URL для проверки
            max_wait: Максимальное время ожидания в секундах
            check_interval: Интервал проверки в секундах
            
        Returns:
            True если сервис стал доступен, False если таймаут
        """
        logger.info(f"Ожидание готовности сервиса {url}...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if self.check_service_health(url):
                elapsed = time.time() - start_time
                logger.info(f"✅ Сервис {url} готов (за {elapsed:.1f}с)")
                return True
            time.sleep(check_interval)
        
        logger.warning(f"⏱️ Таймаут ожидания сервиса {url} ({max_wait}с)")
        return False
    
    def start_with_docker_compose(self, services: List[str]) -> bool:
        """
        Запускает сервисы через docker-compose.
        
        Args:
            services: Список имен сервисов для запуска
            
        Returns:
            True если запуск успешен
        """
        if not self.docker_compose_path.exists():
            logger.error(f"docker-compose.yml не найден: {self.docker_compose_path}")
            return False
        
        try:
            # Запускаем сервисы
            cmd = ["docker-compose", "up", "-d"] + services
            logger.info(f"Запуск сервисов через docker-compose: {', '.join(services)}")
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info("✅ Сервисы запущены через docker-compose")
            self.started_services = services
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при запуске docker-compose: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("docker-compose не найден. Установите Docker Compose.")
            return False
    
    def start_directly(self, service_name: str, module_path: str, port: int) -> bool:
        """
        Запускает сервис напрямую через uvicorn.
        
        Args:
            service_name: Имя сервиса (для логирования)
            module_path: Путь к модулю (например, "tplexity.generation.app:app")
            port: Порт для запуска
            
        Returns:
            True если запуск успешен
        """
        try:
            cmd = [
                sys.executable, "-m", "uvicorn",
                module_path,
                "--host", "0.0.0.0",
                "--port", str(port),
                "--log-level", "info"
            ]
            
            logger.info(f"Запуск {service_name} напрямую на порту {port}...")
            
            # Настраиваем окружение для процесса
            env = os.environ.copy()
            src_path = str(self.project_root / "src")
            if "PYTHONPATH" in env:
                env["PYTHONPATH"] = f"{src_path}:{env['PYTHONPATH']}"
            else:
                env["PYTHONPATH"] = src_path
            
            # Запускаем процесс с выводом логов в файл для отладки
            log_file = self.project_root / "eval" / "outputs" / "logs" / f"{service_name}.log"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Открываем файл в режиме append, чтобы логи накапливались
            log_handle = open(log_file, "a", buffering=1)  # line buffering
            
            process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=env
            )
            
            # Сохраняем handle для последующего закрытия
            self.processes[f"{service_name}_log"] = log_handle
            
            self.processes[service_name] = process
            self.started_services.append(service_name)
            
            # Даем процессу время на запуск
            time.sleep(3)
            
            if process.poll() is None:
                logger.info(f"✅ {service_name} запущен (PID: {process.pid}, логи: {log_file})")
                return True
            else:
                logger.error(f"❌ {service_name} завершился с ошибкой")
                # Читаем последние строки из лог-файла
                try:
                    with open(log_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            logger.error(f"Последние строки логов {service_name}:")
                            for line in lines[-10:]:
                                logger.error(f"  {line.rstrip()}")
                except Exception:
                    pass
                return False
                
        except Exception as e:
            logger.error(f"Ошибка при запуске {service_name}: {e}")
            return False
    
    def stop_service(self, service_name: str):
        """Останавливает конкретный сервис."""
        if service_name in self.processes:
            # Останавливаем процесс напрямую
            process = self.processes[service_name]
            try:
                process.terminate()
                process.wait(timeout=10)
                logger.info(f"✅ {service_name} остановлен")
            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning(f"⚠️ {service_name} принудительно завершен")
            except Exception as e:
                logger.error(f"Ошибка при остановке {service_name}: {e}")
            finally:
                del self.processes[service_name]
                # Закрываем файл логов, если он есть
                log_key = f"{service_name}_log"
                if log_key in self.processes:
                    try:
                        self.processes[log_key].close()
                    except Exception:
                        pass
                    del self.processes[log_key]
        
        elif self.use_docker and service_name in self.started_services:
            # Останавливаем через docker-compose
            try:
                subprocess.run(
                    ["docker-compose", "stop", service_name],
                    cwd=self.project_root,
                    capture_output=True,
                    check=True
                )
                logger.info(f"✅ {service_name} остановлен через docker-compose")
            except Exception as e:
                logger.error(f"Ошибка при остановке {service_name}: {e}")
    
    def stop_all_services(self):
        """Останавливает все запущенные сервисы."""
        if not self.started_services:
            return
        
        logger.info("Остановка всех запущенных сервисов...")
        
        if self.use_docker:
            try:
                subprocess.run(
                    ["docker-compose", "stop"] + self.started_services,
                    cwd=self.project_root,
                    capture_output=True,
                    check=True
                )
                logger.info("✅ Все сервисы остановлены через docker-compose")
            except Exception as e:
                logger.error(f"Ошибка при остановке сервисов: {e}")
        else:
            for service_name in list(self.processes.keys()):
                self.stop_service(service_name)
        
        self.started_services.clear()
    
    def ensure_generation_service(
        self,
        url: str = "http://localhost:8022/health",
        wait: bool = True
    ) -> bool:
        """
        Обеспечивает доступность generation сервиса.
        
        Args:
            url: URL для проверки health
            wait: Ждать готовности сервиса
            
        Returns:
            True если сервис доступен или успешно запущен
        """
        # Проверяем, доступен ли сервис
        if self.check_service_health(url):
            logger.info("✅ Generation сервис уже доступен")
            return True
        
        # Пытаемся запустить
        logger.info("🔄 Generation сервис недоступен, запускаем...")
        
        if self.use_docker:
            success = self.start_with_docker_compose(["generation"])
        else:
            success = self.start_directly(
                "generation",
                "tplexity.generation.app:app",
                8022
            )
        
        if not success:
            logger.error("❌ Не удалось запустить generation сервис")
            return False
        
        # Ждем готовности
        if wait:
            return self.wait_for_service(url)
        
        return True
    
    def ensure_services_for_inference(
        self,
        inference_endpoint: str,
        wait: bool = True
    ) -> bool:
        """
        Обеспечивает доступность всех сервисов, необходимых для inference.
        
        Args:
            inference_endpoint: URL inference endpoint
            wait: Ждать готовности сервисов
            
        Returns:
            True если все сервисы доступны
        """
        # Если endpoint указывает на generation сервис
        if ":8022" in inference_endpoint or "generation" in inference_endpoint.lower():
            # Generation зависит только от retriever (Redis не нужен, т.к. session_id=None)
            services_to_start = ["retriever", "generation"]
            
            # Проверяем каждый сервис
            health_urls = {
                "retriever": "http://localhost:8020/health",
                "generation": "http://localhost:8022/health"
            }
            
            # Проверяем доступность
            need_start = []
            for service in services_to_start:
                if health_urls[service]:
                    if not self.check_service_health(health_urls[service]):
                        need_start.append(service)
                    else:
                        logger.info(f"✅ {service} уже доступен")
            
            if need_start:
                logger.info(f"🔄 Запуск необходимых сервисов: {', '.join(need_start)}")
                
                if self.use_docker:
                    success = self.start_with_docker_compose(need_start)
                else:
                    # Запускаем напрямую
                    success = True
                    
                    # Настраиваем переменные окружения для локального запуска
                    import os
                    os.environ["RETRIEVER_API_URL"] = "http://localhost:8020"
                    os.environ["QWEN_BASE_URL"] = "http://localhost:8100/v1"
                    # Redis не нужен, т.к. session_id=None (память отключена)
                    logger.info(f"Установлены переменные окружения: RETRIEVER_API_URL={os.environ.get('RETRIEVER_API_URL')}, QWEN_BASE_URL={os.environ.get('QWEN_BASE_URL')}")
                    
                    if "retriever" in need_start:
                        success = self.start_directly(
                            "retriever",
                            "tplexity.retriever.app:app",
                            8020
                        ) and success
                    if "generation" in need_start:
                        success = self.start_directly(
                            "generation",
                            "tplexity.generation.app:app",
                            8022
                        ) and success
                
                if not success:
                    return False
                
                # Ждем готовности
                if wait:
                    if "retriever" in need_start:
                        self.wait_for_service("http://localhost:8020/health")
                    if "generation" in need_start:
                        self.wait_for_service("http://localhost:8022/health")
            
            return True
        
        # Для других endpoints просто проверяем доступность
        logger.info(f"Проверка доступности inference endpoint: {inference_endpoint}")
        try:
            response = requests.get(inference_endpoint.replace("/generate", "/health"), timeout=5)
            if response.status_code == 200:
                logger.info("✅ Inference endpoint доступен")
                return True
        except:
            pass
        
        logger.warning(f"⚠️ Inference endpoint {inference_endpoint} недоступен")
        return False

