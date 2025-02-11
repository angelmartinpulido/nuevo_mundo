"""
Advanced resource optimization module for P2P system.
"""

import psutil
import asyncio
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
import cpuinfo
import gputil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


@dataclass
class ResourceMetrics:
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_io: float
    power_consumption: float
    temperature: float


class AdaptiveResourceOptimizer:
    def __init__(self):
        self.cpu_info = cpuinfo.get_cpu_info()
        self.metrics_history: Dict[str, list[ResourceMetrics]] = {}
        self.thread_pool = self._init_thread_pool()
        self.process_pool = self._init_process_pool()
        self.gpu_available = len(gputil.getGPUs()) > 0

    def _init_thread_pool(self) -> ThreadPoolExecutor:
        """Initialize optimized thread pool."""
        cpu_count = psutil.cpu_count(logical=True)
        optimal_threads = self._calculate_optimal_threads()
        return ThreadPoolExecutor(
            max_workers=optimal_threads, thread_name_prefix="optimized_worker"
        )

    def _init_process_pool(self) -> ProcessPoolExecutor:
        """Initialize optimized process pool."""
        cpu_count = psutil.cpu_count(logical=False)
        optimal_processes = max(1, cpu_count - 1)  # Leave one core for system
        return ProcessPoolExecutor(max_workers=optimal_processes)

    def _calculate_optimal_threads(self) -> int:
        """Calculate optimal number of threads based on system capabilities."""
        cpu_count = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()

        # Consider CPU architecture
        if "AMD" in self.cpu_info["brand_raw"]:
            # AMD CPUs often perform better with more threads
            thread_multiplier = 1.5
        else:
            # Intel CPUs might benefit from slightly fewer threads
            thread_multiplier = 1.2

        # Base calculation
        optimal_threads = int(cpu_count * thread_multiplier)

        # Adjust for available memory
        memory_factor = memory.available / (1024 * 1024 * 1024)  # Convert to GB
        memory_threads = int(memory_factor * 2)  # 2 threads per GB

        # Take the minimum to avoid oversubscription
        return min(optimal_threads, memory_threads, cpu_count * 2)

    async def optimize_resources(self) -> Dict[str, Any]:
        """Optimize system resources based on current conditions."""
        metrics = await self._measure_resource_usage()
        self._update_metrics_history(metrics)

        optimizations = {
            "cpu": self._optimize_cpu_usage(metrics),
            "memory": self._optimize_memory_usage(metrics),
            "io": self._optimize_io_operations(metrics),
            "gpu": self._optimize_gpu_usage(metrics) if self.gpu_available else None,
            "power": self._optimize_power_consumption(metrics),
        }

        return optimizations

    async def _measure_resource_usage(self) -> ResourceMetrics:
        """Measure current resource usage with high precision."""
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_io_counters()
        network = psutil.net_io_counters()

        # Get power consumption if available
        try:
            power = (
                psutil.sensors_battery().percent if psutil.sensors_battery() else 100
            )
        except:
            power = 100

        # Get CPU temperature if available
        try:
            temp = psutil.sensors_temperatures()["coretemp"][0].current
        except:
            temp = 0

        return ResourceMetrics(
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_io=disk.read_bytes + disk.write_bytes,
            network_io=network.bytes_sent + network.bytes_recv,
            power_consumption=power,
            temperature=temp,
        )

    def _optimize_cpu_usage(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Advanced CPU optimization strategies."""
        high_load = metrics.cpu_usage > 80
        high_temp = metrics.temperature > 80

        optimizations = {
            "thread_count": self._calculate_optimal_threads(),
            "process_count": max(1, psutil.cpu_count(logical=False) - 1),
            "cpu_governor": "performance" if not high_temp else "powersave",
            "task_priority": {
                "realtime": not high_load,
                "batch_processing": high_load,
                "background": high_load or high_temp,
            },
        }

        if high_temp:
            optimizations.update({"throttle_factor": 0.7, "cooling_period": 5000})  # ms

        return optimizations

    def _optimize_memory_usage(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Advanced memory optimization strategies."""
        memory = psutil.virtual_memory()

        return {
            "cache_size": self._calculate_optimal_cache_size(memory),
            "buffer_size": self._calculate_optimal_buffer_size(memory),
            "gc_threshold": self._calculate_gc_threshold(memory),
            "memory_limit": self._calculate_memory_limit(memory),
            "swap_usage": self._optimize_swap_usage(memory),
        }

    def _optimize_io_operations(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Advanced I/O optimization strategies."""
        return {
            "read_ahead": self._calculate_read_ahead(),
            "write_buffer": self._calculate_write_buffer(),
            "io_scheduler": self._select_io_scheduler(),
            "direct_io": metrics.disk_io
            > 1000000,  # Use direct I/O for high throughput
            "async_io": True,
            "batch_size": self._calculate_io_batch_size(metrics),
        }

    def _optimize_gpu_usage(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Optimize GPU usage if available."""
        if not self.gpu_available:
            return None

        gpus = gputil.getGPUs()
        return {
            "use_gpu": any(gpu.load < 50 for gpu in gpus),
            "gpu_memory_limit": min(gpu.memoryFree for gpu in gpus) * 0.8,
            "cuda_streams": len(gpus) * 2,
            "gpu_batch_size": self._calculate_gpu_batch_size(),
        }

    def _optimize_power_consumption(self, metrics: ResourceMetrics) -> Dict[str, Any]:
        """Advanced power optimization strategies."""
        battery = psutil.sensors_battery()
        on_battery = battery and not battery.power_plugged if battery else False

        return {
            "power_mode": "powersave" if on_battery else "performance",
            "cpu_freq_scaling": {
                "min_freq": "800MHz" if on_battery else "1.2GHz",
                "max_freq": "2.0GHz" if on_battery else "max",
            },
            "core_parking": on_battery,
            "device_power_saving": on_battery,
            "background_tasks": not on_battery,
        }

    def _calculate_optimal_cache_size(self, memory) -> int:
        """Calculate optimal cache size based on available memory."""
        available_mb = memory.available / (1024 * 1024)
        return int(min(available_mb * 0.3, 1024))  # 30% of available memory or 1GB

    def _calculate_optimal_buffer_size(self, memory) -> int:
        """Calculate optimal buffer size for I/O operations."""
        available_mb = memory.available / (1024 * 1024)
        return int(min(available_mb * 0.1, 256))  # 10% of available memory or 256MB

    def _calculate_gc_threshold(self, memory) -> int:
        """Calculate garbage collection threshold."""
        return int(memory.total * 0.75)  # 75% of total memory

    def _calculate_memory_limit(self, memory) -> int:
        """Calculate memory usage limit."""
        return int(memory.total * 0.9)  # 90% of total memory

    def _optimize_swap_usage(self, memory) -> Dict[str, Any]:
        """Optimize swap usage settings."""
        return {
            "swappiness": 60 if memory.percent < 80 else 30,
            "pressure_threshold": 90 if memory.percent < 80 else 70,
            "vfs_cache_pressure": 50 if memory.percent < 80 else 80,
        }

    def _calculate_read_ahead(self) -> int:
        """Calculate optimal read-ahead value."""
        return 256  # KB

    def _calculate_write_buffer(self) -> int:
        """Calculate optimal write buffer size."""
        return 1024  # KB

    def _select_io_scheduler(self) -> str:
        """Select optimal I/O scheduler."""
        return "bfq"  # Budget Fair Queueing

    def _calculate_io_batch_size(self, metrics: ResourceMetrics) -> int:
        """Calculate optimal I/O batch size."""
        if metrics.disk_io > 1000000:
            return 1024 * 1024  # 1MB for high I/O
        return 64 * 1024  # 64KB for normal I/O

    def _calculate_gpu_batch_size(self) -> int:
        """Calculate optimal GPU batch size."""
        gpus = gputil.getGPUs()
        if not gpus:
            return 0

        # Consider GPU memory and compute capability
        free_memory = min(gpu.memoryFree for gpu in gpus)
        return min(int(free_memory * 0.1), 1024)  # 10% of free memory or 1GB

    def _update_metrics_history(self, metrics: ResourceMetrics):
        """Update metrics history for trend analysis."""
        timestamp = asyncio.get_event_loop().time()
        if timestamp not in self.metrics_history:
            self.metrics_history[timestamp] = metrics

        # Keep last hour of metrics
        cutoff = timestamp - 3600
        self.metrics_history = {
            k: v for k, v in self.metrics_history.items() if k > cutoff
        }

    def analyze_resource_trends(self) -> Dict[str, Any]:
        """Analyze resource usage trends for predictive optimization."""
        if not self.metrics_history:
            return {}

        timestamps = list(self.metrics_history.keys())
        metrics = list(self.metrics_history.values())

        return {
            "cpu_trend": np.polyfit(timestamps, [m.cpu_usage for m in metrics], 1)[0],
            "memory_trend": np.polyfit(
                timestamps, [m.memory_usage for m in metrics], 1
            )[0],
            "io_trend": np.polyfit(timestamps, [m.disk_io for m in metrics], 1)[0],
            "temperature_trend": np.polyfit(
                timestamps, [m.temperature for m in metrics], 1
            )[0],
        }
