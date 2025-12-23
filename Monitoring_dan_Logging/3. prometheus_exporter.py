from prometheus_client import Gauge, start_http_server
import psutil
import time

EXPORTER_CPU = Gauge("exporter_cpu_percent", "CPU usage percent from exporter")
EXPORTER_MEMORY = Gauge("exporter_memory_percent", "Memory usage percent from exporter")

PORT = 9000

if __name__ == "__main__":
    start_http_server(PORT)
    print(f"Exporter on port {PORT}")

    while True:
        EXPORTER_CPU.set(psutil.cpu_percent())
        EXPORTER_MEMORY.set(psutil.virtual_memory().percent)
        time.sleep(5)