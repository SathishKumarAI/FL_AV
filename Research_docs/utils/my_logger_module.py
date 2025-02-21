import logging
import os
from pathlib import Path
import atexit
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
import smtplib
import threading
import sqlite3
from datetime import datetime, timedelta
from zipfile import ZipFile


class EmailAlertHandler(logging.Handler):
    """Custom handler to send email alerts for CRITICAL logs."""
    def __init__(self, email_alerts):
        super().__init__()
        self.email_alerts = email_alerts

    def emit(self, record):
        """Send an email alert for CRITICAL logs."""
        if self.email_alerts and record.levelno == logging.CRITICAL:
            self.send_email_alert(self.format(record))

    def send_email_alert(self, message):
        """Send an email using SMTP."""
        try:
            with smtplib.SMTP("smtp.example.com", 587) as server:
                server.starttls()
                server.login("your_email@example.com", "password")
                server.sendmail(
                    "your_email@example.com",
                    "alert_recipient@example.com",
                    f"Subject: Critical Error Alert\n\n{message}"
                )
            print("📩 Email alert sent!")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")


class LoggerManager:
    """
    Manages logging for scripts, offering:
    - File and console logging with rotating log files
    - Dynamic log level setting (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Email alerts for critical issues
    - Log searching and old log cleanup
    - Optional logging to SQLite database
    """

    _instances = {}  # Singleton instance tracking

    def __new__(cls, notebook_name, category="general", max_bytes=5*(1024**2), backup_count=3, 
                email_alerts=False, log_to_db=False):
        key = f"{category}/{notebook_name}"
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance.initialized = False
            cls._instances[key] = instance
        return cls._instances[key]

    def __init__(self, notebook_name, category="general", max_bytes=5*(1024**2), backup_count=3, 
                email_alerts=False, log_to_db=False):
        if self.initialized:
            return

        self.notebook_name = notebook_name
        self.category = category
        self.email_alerts = email_alerts
        self.log_to_db = log_to_db
        self.db_lock = threading.Lock()  # Thread safety for database operations

        # Set up log directory
        base_log_dir = Path(r"C:\Users\devil\Downloads\FL_AV\Research_docs\logs")
        self.base_log_dir = base_log_dir / category / notebook_name
        # self.base_log_dir.mkdir(parents=True, exist_ok=True)
        # self.base_log_dir = Path.cwd() / "logs" / category / notebook_name
        self.base_log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_file = self.base_log_dir / f"log_{timestamp}.log"

        # Initialize logger
        self.logger = logging.getLogger(f"{category}/{notebook_name}")
        self.logger.setLevel(logging.DEBUG)  # Default level to DEBUG

        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Setup handlers and cleanup
        self._setup_handlers()
        self._setup_cleanup_task()

        # Setup database if enabled
        if log_to_db:
            self._setup_db()

        # Ensure logger is closed on program exit
        atexit.register(self.close_logger)

        print(f"✅ Logging initialized for '{notebook_name}' in category '{category}'. Logs saved in {self.log_file}")
        self.initialized = True

    def _setup_handlers(self):
        """Sets up file logging, console output, and email alerts."""
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # File handler with log rotation
        file_handler = RotatingFileHandler(
            self.log_file, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        # Console handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        # Email alert handler for critical logs
        if self.email_alerts:
            email_handler = EmailAlertHandler(self.email_alerts)
            email_handler.setLevel(logging.CRITICAL)
            self.logger.addHandler(email_handler)

        print(f"✅ Logger handlers initialized. Log file: {self.log_file}")

    def _setup_cleanup_task(self):
        """Cleans up old logs asynchronously."""
        cleanup_thread = threading.Thread(target=self._clean_old_logs, daemon=True)
        cleanup_thread.start()

    def _clean_old_logs(self, days=7):
        """Deletes logs older than X days and compresses them into ZIP files."""
        print("🧹 Checking for old logs to clean...")
        for file in self.base_log_dir.glob("log_*.log"):
            try:
                timestamp_str = "_".join(file.stem.split("_")[1:])
                file_time = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                print(f"⚠️ Skipping malformed log filename: {file}")
                continue

            if datetime.now() - file_time > timedelta(days=days):
                zip_path = file.with_suffix(".zip")
                with ZipFile(zip_path, "w") as zipf:
                    zipf.write(file, file.name)
                file.unlink()
                print(f"🗑️ Old log {file} compressed to {zip_path}")

    def _setup_db(self):
        """Sets up SQLite database for logs."""
        self.db_path = self.base_log_dir / "logs.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                timestamp TEXT, 
                level TEXT, 
                message TEXT
            )
        """)
        self.conn.commit()

    def log(self, level, message):
        """Logs a message and optionally stores it in the database."""
        self.logger.log(level, message)
        if self.log_to_db:
            self._log_to_db(level, message)

    def _log_to_db(self, level, message):
        """Logs data into the database."""
        with self.db_lock:
            self.cursor.execute("INSERT INTO logs (timestamp, level, message) VALUES (?, ?, ?)", 
                                (datetime.now().isoformat(), level, message))
            self.conn.commit()

    def search_logs(self, keyword):
        """Search logs for a specific keyword."""
        if not self.log_file.exists():
            print("❌ No log file found.")
            return []

        with open(self.log_file, "r", encoding="utf-8") as log_fh:
            matches = [line.strip() for line in log_fh if keyword.lower() in line.lower()]

        print(f"✅ Found {len(matches)} matches for '{keyword}'") if matches else print(f"🔍 No matches found.")

        return matches

    def set_log_level(self, level):
        """Dynamically adjust the log level."""
        level_mapping = {
            "DEBUG": logging.DEBUG, "INFO": logging.INFO, 
            "WARNING": logging.WARNING, "ERROR": logging.ERROR, 
            "CRITICAL": logging.CRITICAL
        }
        if level.upper() in level_mapping:
            self.logger.setLevel(level_mapping[level.upper()])
            print(f"✅ Log level set to {level.upper()}")
        else:
            print("❌ Invalid log level.")

    def close_logger(self):
        """Closes all handlers and database connections."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        if self.log_to_db:
            self.conn.close()
        print(f"✅ Logger '{self.notebook_name}' closed successfully.")

    @contextmanager
    def temporary_logger(self):
        """Context manager for temporary logging."""
        try:
            yield self.logger
        finally:
            self.close_logger()
