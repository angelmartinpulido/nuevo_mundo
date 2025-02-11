"""
Adaptive optimization system that continuously monitors and adjusts P2P performance.
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
import logging
from .network_optimizer import AdaptiveNetworkOptimizer
from .resource_optimizer import AdaptiveResourceOptimizer


@dataclass
class PerformanceMetrics:
    timestamp: float
    latency: float
    throughput: float
    cpu_usage: float
    memory_usage: float
    power_usage: float
    error_rate: float
    connection_count: int


class AdaptiveOptimizer:
    def __init__(self):
        self.network_optimizer = AdaptiveNetworkOptimizer()
        self.resource_optimizer = AdaptiveResourceOptimizer()
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_state = {}
        self.learning_rate = 0.01
        self.exploration_rate = 0.1

    async def start_optimization_loop(self):
        """Start continuous optimization loop."""
        while True:
            try:
                # Collect current metrics
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)

                # Analyze trends
                trends = self._analyze_trends()

                # Determine optimization actions
                actions = self._determine_optimization_actions(trends)

                # Apply optimizations
                await self._apply_optimizations(actions)

                # Evaluate results
                impact = await self._evaluate_optimization_impact()

                # Update learning parameters
                self._update_learning_parameters(impact)

                # Clean up old metrics
                self._cleanup_old_metrics()

                # Wait before next optimization cycle
                await asyncio.sleep(60)  # Adjust based on system needs

            except Exception as e:
                logging.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)  # Back off on error

    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        # Network metrics
        network_stats = await self.network_optimizer._measure_network_conditions(
            "global"
        )

        # Resource metrics
        resource_metrics = await self.resource_optimizer._measure_resource_usage()

        # System metrics
        system_metrics = self._collect_system_metrics()

        return PerformanceMetrics(
            timestamp=time.time(),
            latency=network_stats.rtt,
            throughput=network_stats.bandwidth,
            cpu_usage=resource_metrics.cpu_usage,
            memory_usage=resource_metrics.memory_usage,
            power_usage=resource_metrics.power_consumption,
            error_rate=system_metrics["error_rate"],
            connection_count=system_metrics["connection_count"],
        )

    def _analyze_trends(self) -> Dict[str, float]:
        """Analyze performance trends using advanced statistical methods."""
        if len(self.metrics_history) < 2:
            return {}

        metrics_array = np.array(
            [
                [
                    m.latency,
                    m.throughput,
                    m.cpu_usage,
                    m.memory_usage,
                    m.power_usage,
                    m.error_rate,
                    m.connection_count,
                ]
                for m in self.metrics_history
            ]
        )

        # Calculate trends using weighted linear regression
        weights = np.exp(np.linspace(-1, 0, len(self.metrics_history)))
        trends = {}

        for i, metric_name in enumerate(
            [
                "latency",
                "throughput",
                "cpu_usage",
                "memory_usage",
                "power_usage",
                "error_rate",
                "connection_count",
            ]
        ):
            trend = np.polyfit(
                range(len(self.metrics_history)), metrics_array[:, i], deg=1, w=weights
            )[0]
            trends[metric_name] = trend

        return trends

    def _determine_optimization_actions(
        self, trends: Dict[str, float]
    ) -> Dict[str, Any]:
        """Determine necessary optimization actions based on trends."""
        actions = {}

        # Network optimizations
        if trends.get("latency", 0) > 0:  # Latency increasing
            actions["network"] = {
                "protocol": "quic" if trends["error_rate"] < 0.01 else "tcp",
                "compression_level": min(9, int(trends["latency"])),
                "batch_size": self._optimize_batch_size(trends),
            }

        # Resource optimizations
        if trends.get("cpu_usage", 0) > 0:  # CPU usage increasing
            actions["cpu"] = {
                "thread_count": self._optimize_thread_count(trends),
                "process_count": self._optimize_process_count(trends),
            }

        if trends.get("memory_usage", 0) > 0:  # Memory usage increasing
            actions["memory"] = {
                "cache_size": self._optimize_cache_size(trends),
                "gc_threshold": self._optimize_gc_threshold(trends),
            }

        # Power optimizations
        if trends.get("power_usage", 0) > 0:  # Power usage increasing
            actions["power"] = {
                "power_mode": "powersave",
                "cpu_freq": self._optimize_cpu_frequency(trends),
            }

        return actions

    async def _apply_optimizations(self, actions: Dict[str, Any]):
        """Apply optimization actions with safety checks."""
        try:
            # Network optimizations
            if "network" in actions:
                network_opts = await self.network_optimizer.optimize_connection(
                    "global", actions["network"]
                )
                self.optimization_state["network"] = network_opts

            # Resource optimizations
            if "cpu" in actions or "memory" in actions:
                resource_opts = await self.resource_optimizer.optimize_resources()
                self.optimization_state["resources"] = resource_opts

            # Power optimizations
            if "power" in actions:
                await self._apply_power_optimizations(actions["power"])

            # Record applied optimizations
            self.optimization_state["last_applied"] = time.time()
            self.optimization_state["actions"] = actions

        except Exception as e:
            logging.error(f"Error applying optimizations: {e}")
            # Rollback if necessary
            await self._rollback_optimizations()

    async def _evaluate_optimization_impact(self) -> float:
        """Evaluate the impact of applied optimizations."""
        if not self.optimization_state.get("last_applied"):
            return 0.0

        # Collect metrics after optimization
        current_metrics = await self._collect_metrics()

        # Compare with previous metrics
        if len(self.metrics_history) < 2:
            return 0.0

        previous_metrics = self.metrics_history[-2]

        # Calculate improvement scores
        improvements = {
            "latency": (previous_metrics.latency - current_metrics.latency)
            / previous_metrics.latency,
            "throughput": (current_metrics.throughput - previous_metrics.throughput)
            / previous_metrics.throughput,
            "cpu_usage": (previous_metrics.cpu_usage - current_metrics.cpu_usage)
            / previous_metrics.cpu_usage,
            "memory_usage": (
                previous_metrics.memory_usage - current_metrics.memory_usage
            )
            / previous_metrics.memory_usage,
            "power_usage": (previous_metrics.power_usage - current_metrics.power_usage)
            / previous_metrics.power_usage,
            "error_rate": (previous_metrics.error_rate - current_metrics.error_rate)
            / (previous_metrics.error_rate + 1e-6),
        }

        # Calculate weighted impact score
        weights = {
            "latency": 0.25,
            "throughput": 0.25,
            "cpu_usage": 0.15,
            "memory_usage": 0.15,
            "power_usage": 0.1,
            "error_rate": 0.1,
        }

        impact_score = sum(improvements[k] * weights[k] for k in weights)
        return impact_score

    def _update_learning_parameters(self, impact: float):
        """Update learning parameters based on optimization impact."""
        # Adjust learning rate
        if impact > 0:
            self.learning_rate *= 1.1  # Increase learning rate for positive impact
        else:
            self.learning_rate *= 0.9  # Decrease learning rate for negative impact

        # Bound learning rate
        self.learning_rate = max(0.001, min(0.1, self.learning_rate))

        # Adjust exploration rate
        self.exploration_rate *= 0.995  # Gradually reduce exploration
        self.exploration_rate = max(
            0.01, self.exploration_rate
        )  # Maintain minimum exploration

    def _cleanup_old_metrics(self):
        """Clean up old metrics to prevent memory bloat."""
        # Keep last 24 hours of metrics
        cutoff_time = time.time() - (24 * 3600)
        self.metrics_history = [
            m for m in self.metrics_history if m.timestamp > cutoff_time
        ]

    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-wide metrics."""
        return {
            "error_rate": self._calculate_error_rate(),
            "connection_count": self._get_connection_count(),
        }

    def _calculate_error_rate(self) -> float:
        """Calculate current error rate."""
        # Implementation of error rate calculation
        return 0.0  # Placeholder

    def _get_connection_count(self) -> int:
        """Get current connection count."""
        # Implementation of connection counting
        return 0  # Placeholder

    def _optimize_batch_size(self, trends: Dict[str, float]) -> int:
        """Calculate optimal batch size based on trends."""
        base_size = 1024  # 1KB base size

        # Adjust based on latency trend
        latency_factor = 1 + abs(trends.get("latency", 0))

        # Adjust based on throughput trend
        throughput_factor = 1 + trends.get("throughput", 0)

        optimal_size = int(base_size * latency_factor * throughput_factor)
        return max(1024, min(optimal_size, 1048576))  # Between 1KB and 1MB

    def _optimize_thread_count(self, trends: Dict[str, float]) -> int:
        """Calculate optimal thread count based on trends."""
        current_threads = (
            self.optimization_state.get("resources", {})
            .get("cpu", {})
            .get("thread_count", 0)
        )

        if trends["cpu_usage"] > 0:
            # CPU usage increasing, reduce threads
            return max(1, int(current_threads * 0.8))
        else:
            # CPU usage decreasing, can increase threads
            return min(int(current_threads * 1.2), psutil.cpu_count() * 2)

    def _optimize_process_count(self, trends: Dict[str, float]) -> int:
        """Calculate optimal process count based on trends."""
        cpu_count = psutil.cpu_count(logical=False)

        if trends["cpu_usage"] > 0:
            # High CPU usage, be conservative
            return max(1, cpu_count - 1)
        else:
            # Low CPU usage, can use more processes
            return cpu_count

    def _optimize_cache_size(self, trends: Dict[str, float]) -> int:
        """Calculate optimal cache size based on trends."""
        current_cache = (
            self.optimization_state.get("resources", {})
            .get("memory", {})
            .get("cache_size", 0)
        )

        if trends["memory_usage"] > 0:
            # Memory usage increasing, reduce cache
            return max(1024, int(current_cache * 0.8))
        else:
            # Memory usage decreasing, can increase cache
            return min(int(current_cache * 1.2), self._get_max_cache_size())

    def _optimize_gc_threshold(self, trends: Dict[str, float]) -> int:
        """Calculate optimal garbage collection threshold."""
        base_threshold = 75  # 75% base threshold

        if trends["memory_usage"] > 0:
            # Memory pressure increasing
            return max(60, base_threshold - 10)
        else:
            # Memory pressure decreasing
            return min(90, base_threshold + 10)

    def _optimize_cpu_frequency(self, trends: Dict[str, float]) -> str:
        """Calculate optimal CPU frequency based on trends."""
        if trends["power_usage"] > 0:
            # Power usage increasing
            return "powersave"
        elif trends["cpu_usage"] > 0:
            # High CPU demand
            return "performance"
        else:
            # Balanced
            return "ondemand"

    def _get_max_cache_size(self) -> int:
        """Get maximum allowed cache size based on system memory."""
        total_memory = psutil.virtual_memory().total
        return int(total_memory * 0.2)  # 20% of total memory

    async def _rollback_optimizations(self):
        """Rollback optimizations in case of failure."""
        if "previous_state" in self.optimization_state:
            self.optimization_state = self.optimization_state["previous_state"]
        else:
            # Reset to default values
            self.optimization_state = {}
