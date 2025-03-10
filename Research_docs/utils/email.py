import logging
from logging.handlers import RotatingFileHandler
import smtplib


class EmailAlertHandler(logging.Handler):
    def __init__(self, email_alerts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.email_alerts = email_alerts

    def emit(self, record):
        if self.email_alerts and record.levelno == logging.CRITICAL:
            self.send_email_alert(self.format(record))

    def send_email_alert(self, message):
        try:
            server = smtplib.SMTP("smtp.example.com", 587)
            server.starttls()
            server.login("your_email@example.com", "password")
            server.sendmail(
                "your_email@example.com",
                "alert_recipient@example.com",
                f"Subject: Critical Error Alert\n\n{message}"
            )
            server.quit()
            print("📩 Email alert sent!")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")

def _setup_handlers(self):
    """Sets up Rotating File Handler, Console Output, and Email Alerts."""
    if self.logger.hasHandlers():
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    file_handler = RotatingFileHandler(
        self.log_file, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    self.logger.addHandler(file_handler)
    self.logger.addHandler(stream_handler)

    if self.email_alerts:
        email_handler = EmailAlertHandler(self.email_alerts)
        email_handler.setLevel(logging.CRITICAL)
        self.logger.addHandler(email_handler)