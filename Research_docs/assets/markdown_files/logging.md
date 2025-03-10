Below is a **fully-optimized** Python logging class that integrates all of the best practices and enhancements discussed. It uses object-oriented design, the built-in `logging` module, rotating file handlers, console toggling, dynamic log-level configuration, and more. 

---


## How to Use This Class

1. **Initialize the Logger in Your Notebook or Script**

   ```python
   from my_logger_module import LoggerManager  # Suppose you saved the above class in my_logger_module.py

   # Create a logger manager for a specific notebook/task
   logger_manager = LoggerManager("train_yolo_conversion")

   # Access the logger instance
   logger = logger_manager.logger

   # Log messages at different severity levels
   logger.debug("Debug message (not visible at INFO level).")
   logger.info("Training has started.")
   logger.warning("Potential issue with dataset size.")
   logger.error("An error occurred during training!")
   logger.critical("Critical issue - out of memory!")
   ```

2. **Change Log Level on the Fly**

   ```python
   logger_manager.set_log_level("DEBUG")
   # Now DEBUG messages will also be shown
   logger.debug("This debug message is now visible.")
   ```

3. **Toggle Console (Notebook) Output**

   ```python
   # Turn off console logs (still logs to file)
   logger_manager.enable_console_logging(False)

   # Turn console logs back on
   logger_manager.enable_console_logging(True)
   ```

4. **Update the Log Directory (at Runtime)**

   ```python
   logger_manager.set_log_directory("C:/new/logs/folder")
   # A new timestamped log file is created in the new directory.
   ```

5. **Search Log File for Keywords**

   ```python
   matches = logger_manager.search_logs("ERROR")
   # Prints all lines containing "ERROR" in the current log file
   ```

6. **Close the Logger Manually (Optional)**

   ```python
   logger_manager.close_logger()
   # Automatically done at exit, but you can do it sooner if needed.
   ```

7. **Using a Temporary Logger Session**

   ```python
   # If you just need logging in a short block and want it to auto-close:
   with logger_manager.temporary_logger() as temp_log:
       temp_log.info("Starting a short process...")
       temp_log.warning("This process might fail.")
       # The logger closes automatically at the end of this block.
   ```

---

## Key Benefits & Design Choices

- **Timestamped Log Files**  
  Ensures each new run creates a fresh log file without overwriting old logs.

- **Rotating File Handler**  
  Prevents log files from growing uncontrollably. Once the file size exceeds `max_bytes` (default ~5MB), it rotates, keeping up to `backup_count` backups.

- **Console + File Logging**  
  You get immediate log feedback in your notebook/terminal, and a persistent record stored on disk.

- **Singleton Pattern**  
  Only one `LoggerManager` instance per `notebook_name`, preventing duplicate loggers or repeated handlers.

- **`atexit` Cleanup**  
  Ensures the log file is closed automatically when your script or notebook finishes.

- **Context Manager for Temporary Logging**  
  A convenient pattern for short operations that need isolation, automatically closing the logger afterward.

---

**That’s it!** This single class `LoggerManager` collects all the best logging practices for Python projects, especially in **Jupyter Notebooks** or any environment requiring rotating logs, multiple log levels, console toggling, and flexible, well-structured logging.