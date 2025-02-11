"""
Universal Data Collector Core Module
Provides core functionality for collecting data from any source
"""

import asyncio
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import aiohttp
import logging


class DataSource(ABC):
    """Abstract base class for all data sources"""

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to the data source"""
        pass

    @abstractmethod
    async def collect(self) -> Dict[str, Any]:
        """Collect data from the source"""
        pass

    @abstractmethod
    async def validate(self, data: Dict[str, Any]) -> bool:
        """Validate collected data"""
        pass


class UniversalCollector:
    """Main collector class that orchestrates data collection from all sources"""

    def __init__(self):
        self.sources: Dict[str, DataSource] = {}
        self.collectors: Dict[str, Any] = {}
        self.data_processors: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    async def register_source(self, source_id: str, source: DataSource) -> bool:
        """Register a new data source"""
        try:
            self.sources[source_id] = source
            return True
        except Exception as e:
            self.logger.error(f"Error registering source {source_id}: {str(e)}")
            return False

    async def collect_from_all(self) -> Dict[str, Any]:
        """Collect data from all registered sources"""
        results = {}
        tasks = []

        for source_id, source in self.sources.items():
            tasks.append(self._collect_from_source(source_id, source))

        collected_data = await asyncio.gather(*tasks, return_exceptions=True)

        for source_id, data in zip(self.sources.keys(), collected_data):
            if not isinstance(data, Exception):
                results[source_id] = data
            else:
                self.logger.error(f"Error collecting from {source_id}: {str(data)}")

        return results

    async def _collect_from_source(
        self, source_id: str, source: DataSource
    ) -> Dict[str, Any]:
        """Collect data from a specific source"""
        try:
            if await source.connect():
                data = await source.collect()
                if await source.validate(data):
                    return data
            raise Exception(f"Failed to collect valid data from {source_id}")
        except Exception as e:
            raise Exception(f"Collection error for {source_id}: {str(e)}")


class DataProcessor:
    """Processes and normalizes collected data"""

    def __init__(self):
        self.processors = {}
        self.validators = {}

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process collected data into a standardized format"""
        processed_data = {}

        for source_id, source_data in data.items():
            if source_id in self.processors:
                try:
                    processed = await self.processors[source_id](source_data)
                    if await self._validate_processed(source_id, processed):
                        processed_data[source_id] = processed
                except Exception as e:
                    logging.error(f"Processing error for {source_id}: {str(e)}")

        return processed_data

    async def _validate_processed(self, source_id: str, data: Dict[str, Any]) -> bool:
        """Validate processed data"""
        if source_id in self.validators:
            return await self.validators[source_id](data)
        return True


class DataAdapter:
    """Adapts collected data for storage and distribution"""

    def __init__(self):
        self.adapters = {}

    async def adapt(self, data: Dict[str, Any], target_format: str) -> Dict[str, Any]:
        """Adapt data to specified format"""
        if target_format in self.adapters:
            return await self.adapters[target_format](data)
        return data
