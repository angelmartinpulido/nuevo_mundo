"""
Core fragmentation manager for P2P system.
Handles software fragmentation, distribution, and communication.
"""

import asyncio
import hashlib
import os
import sys
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum
import aiofiles
import psutil
from cryptography.fernet import Fernet
from ...utils.security import SecurityManager


class FragmentPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class FragmentType(Enum):
    CORE = 0
    NETWORK = 1
    STORAGE = 2
    COMPUTE = 3
    SECURITY = 4
    MONITORING = 5


@dataclass
class Fragment:
    id: str
    type: FragmentType
    priority: FragmentPriority
    code: bytes
    dependencies: Set[str]
    state: Dict[str, Any]
    signature: bytes
    version: int
    last_sync: float
    health: float


class FragmentManager:
    def __init__(self):
        self.fragments: Dict[str, Fragment] = {}
        self.active_fragments: Set[str] = set()
        self.fragment_locations: Dict[str, List[str]] = {}
        self.security_manager = SecurityManager()
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.system_monitor = SystemResourceMonitor()

    async def initialize(self):
        """Initialize the fragmentation system."""
        await self._analyze_system()
        await self._create_initial_fragments()
        await self._distribute_fragments()
        await self._start_monitoring()

    async def _analyze_system(self):
        """Analyze system for safe fragmentation."""
        self.system_info = {
            "os": os.name,
            "platform": sys.platform,
            "cpu_count": psutil.cpu_count(),
            "memory": psutil.virtual_memory().total,
            "disk_space": psutil.disk_usage("/").total,
            "security_level": await self._determine_security_level(),
        }

    async def _determine_security_level(self) -> int:
        """Determine system security level."""
        security_score = 0

        # Check OS security features
        if sys.platform.startswith("linux"):
            security_score += await self._check_linux_security()
        elif sys.platform == "win32":
            security_score += await self._check_windows_security()
        elif sys.platform == "darwin":
            security_score += await self._check_macos_security()

        return security_score

    async def create_fragment(
        self, code: bytes, fragment_type: FragmentType, priority: FragmentPriority
    ) -> str:
        """Create a new code fragment."""
        # Validate code safety
        if not await self._validate_code_safety(code):
            raise SecurityError("Code validation failed")

        # Create fragment ID
        fragment_id = self._generate_fragment_id(code)

        # Encrypt code
        encrypted_code = self.fernet.encrypt(code)

        # Create fragment
        fragment = Fragment(
            id=fragment_id,
            type=fragment_type,
            priority=priority,
            code=encrypted_code,
            dependencies=set(),
            state={},
            signature=await self._sign_fragment(encrypted_code),
            version=1,
            last_sync=asyncio.get_event_loop().time(),
            health=1.0,
        )

        # Store fragment
        self.fragments[fragment_id] = fragment
        await self._store_fragment(fragment)

        return fragment_id

    async def distribute_fragment(
        self, fragment_id: str, target_layers: List[str]
    ) -> bool:
        """Distribute fragment across system layers."""
        fragment = self.fragments.get(fragment_id)
        if not fragment:
            return False

        # Check system resource availability
        if not await self.system_monitor.check_resources():
            return False

        # Determine safe storage locations
        safe_locations = await self._find_safe_locations(fragment, target_layers)

        # Distribute copies
        for location in safe_locations:
            success = await self._store_fragment_copy(fragment, location)
            if success:
                self.fragment_locations.setdefault(fragment_id, []).append(location)

        return len(self.fragment_locations.get(fragment_id, [])) > 0

    async def activate_fragment(self, fragment_id: str) -> bool:
        """Activate a fragment for execution."""
        fragment = self.fragments.get(fragment_id)
        if not fragment:
            return False

        # Verify fragment integrity
        if not await self._verify_fragment(fragment):
            return False

        # Check dependencies
        if not await self._check_dependencies(fragment):
            return False

        # Prepare execution environment
        env = await self._prepare_execution_environment(fragment)

        try:
            # Decrypt code
            decrypted_code = self.fernet.decrypt(fragment.code)

            # Execute in isolated environment
            success = await self._execute_fragment(decrypted_code, env)

            if success:
                self.active_fragments.add(fragment_id)
                return True

        except Exception as e:
            logging.error(f"Fragment activation failed: {e}")

        return False

    async def sync_fragments(self):
        """Synchronize fragments across the network."""
        for fragment_id, fragment in self.fragments.items():
            # Check if sync is needed
            if not await self._needs_sync(fragment):
                continue

            # Get fragment copies
            copies = await self._get_fragment_copies(fragment_id)

            # Verify and sync copies
            for copy in copies:
                if not await self._verify_fragment(copy):
                    await self._repair_fragment(copy)

            # Update sync time
            fragment.last_sync = asyncio.get_event_loop().time()

    async def monitor_fragments(self):
        """Monitor health and status of fragments."""
        while True:
            for fragment_id, fragment in self.fragments.items():
                # Check fragment health
                health = await self._check_fragment_health(fragment)
                fragment.health = health

                # Handle unhealthy fragments
                if health < 0.5:
                    await self._repair_fragment(fragment)

                # Check for inactive fragments
                if fragment_id not in self.active_fragments:
                    await self._handle_inactive_fragment(fragment)

            await asyncio.sleep(60)  # Check every minute

    async def _validate_code_safety(self, code: bytes) -> bool:
        """Validate code safety."""
        # Check for dangerous operations
        dangerous_patterns = [
            b"system(",
            b"exec(",
            b"eval(",
            b"subprocess",
            b"os.system",
        ]

        for pattern in dangerous_patterns:
            if pattern in code:
                return False

        # Analyze code structure
        try:
            compile(code, "<string>", "exec")
        except:
            return False

        return True

    async def _find_safe_locations(
        self, fragment: Fragment, target_layers: List[str]
    ) -> List[str]:
        """Find safe storage locations for fragment."""
        safe_locations = []

        for layer in target_layers:
            # Check layer security
            if await self._is_layer_secure(layer):
                # Check layer capacity
                if await self._check_layer_capacity(layer):
                    # Check layer compatibility
                    if await self._check_layer_compatibility(layer, fragment):
                        safe_locations.append(layer)

        # Prioritize locations based on fragment priority
        return self._prioritize_locations(safe_locations, fragment.priority)

    async def _store_fragment_copy(self, fragment: Fragment, location: str) -> bool:
        """Store a copy of fragment in specified location."""
        try:
            # Prepare storage location
            await self._prepare_storage_location(location)

            # Create encrypted container
            container = await self._create_secure_container(fragment, location)

            # Store fragment
            async with aiofiles.open(f"{location}/fragment_{fragment.id}", "wb") as f:
                await f.write(container)

            return True

        except Exception as e:
            logging.error(f"Failed to store fragment copy: {e}")
            return False

    async def _verify_fragment(self, fragment: Fragment) -> bool:
        """Verify fragment integrity and authenticity."""
        # Verify signature
        if not await self.security_manager.verify_signature(
            fragment.code, fragment.signature
        ):
            return False

        # Check code integrity
        if not await self._verify_code_integrity(fragment):
            return False

        # Verify dependencies
        if not await self._verify_dependencies(fragment):
            return False

        return True

    async def _repair_fragment(self, fragment: Fragment):
        """Repair or restore damaged fragment."""
        # Get healthy copies
        healthy_copies = await self._get_healthy_copies(fragment)

        if healthy_copies:
            # Restore from healthy copy
            await self._restore_fragment(fragment, healthy_copies[0])
        else:
            # Recreate fragment
            await self._recreate_fragment(fragment)

    async def _handle_inactive_fragment(self, fragment: Fragment):
        """Handle inactive fragment."""
        # Check if fragment should be active
        if await self._should_be_active(fragment):
            # Attempt to activate
            await self.activate_fragment(fragment.id)
        else:
            # Update fragment state
            fragment.state["inactive_time"] = asyncio.get_event_loop().time()

    def _generate_fragment_id(self, code: bytes) -> str:
        """Generate unique fragment ID."""
        return hashlib.sha256(code).hexdigest()[:16]

    async def _sign_fragment(self, code: bytes) -> bytes:
        """Sign fragment code."""
        return await self.security_manager.sign_data(code)

    async def _check_dependencies(self, fragment: Fragment) -> bool:
        """Check if all dependencies are available and active."""
        for dep_id in fragment.dependencies:
            if dep_id not in self.active_fragments:
                return False
        return True

    async def _prepare_execution_environment(
        self, fragment: Fragment
    ) -> Dict[str, Any]:
        """Prepare isolated execution environment."""
        return {
            "fragment_id": fragment.id,
            "type": fragment.type,
            "state": fragment.state.copy(),
            "security_context": await self._create_security_context(fragment),
        }

    async def _execute_fragment(self, code: bytes, env: Dict[str, Any]) -> bool:
        """Execute fragment code in isolated environment."""
        try:
            # Create isolated namespace
            namespace = {"env": env}

            # Execute code
            exec(code, namespace)
            return True

        except Exception as e:
            logging.error(f"Fragment execution failed: {e}")
            return False

    async def _needs_sync(self, fragment: Fragment) -> bool:
        """Check if fragment needs synchronization."""
        return (asyncio.get_event_loop().time() - fragment.last_sync) > 3600  # 1 hour

    async def _get_fragment_copies(self, fragment_id: str) -> List[Fragment]:
        """Get all copies of a fragment."""
        copies = []
        locations = self.fragment_locations.get(fragment_id, [])

        for location in locations:
            try:
                copy = await self._load_fragment_copy(fragment_id, location)
                copies.append(copy)
            except Exception as e:
                logging.error(f"Failed to load fragment copy: {e}")

        return copies

    async def _check_fragment_health(self, fragment: Fragment) -> float:
        """Check fragment health status."""
        health_score = 1.0

        # Check code integrity
        if not await self._verify_code_integrity(fragment):
            health_score -= 0.3

        # Check dependencies health
        dep_health = await self._check_dependencies_health(fragment)
        health_score *= dep_health

        # Check execution status
        if fragment.id in self.active_fragments:
            exec_health = await self._check_execution_health(fragment)
            health_score *= exec_health

        return max(0.0, min(1.0, health_score))

    async def _verify_code_integrity(self, fragment: Fragment) -> bool:
        """Verify integrity of fragment code."""
        try:
            # Decrypt code
            decrypted_code = self.fernet.decrypt(fragment.code)

            # Verify signature
            return await self.security_manager.verify_signature(
                decrypted_code, fragment.signature
            )

        except Exception:
            return False

    async def _verify_dependencies(self, fragment: Fragment) -> bool:
        """Verify all fragment dependencies."""
        for dep_id in fragment.dependencies:
            dep = self.fragments.get(dep_id)
            if not dep or not await self._verify_fragment(dep):
                return False
        return True

    async def _get_healthy_copies(self, fragment: Fragment) -> List[Fragment]:
        """Get healthy copies of a fragment."""
        copies = await self._get_fragment_copies(fragment.id)
        return [copy for copy in copies if await self._verify_fragment(copy)]

    async def _restore_fragment(self, fragment: Fragment, healthy_copy: Fragment):
        """Restore fragment from healthy copy."""
        fragment.code = healthy_copy.code
        fragment.signature = healthy_copy.signature
        fragment.version = healthy_copy.version
        fragment.state = healthy_copy.state.copy()
        fragment.health = 1.0

    async def _recreate_fragment(self, fragment: Fragment):
        """Recreate damaged fragment."""
        # Get original code template
        template = await self._get_fragment_template(fragment.type)

        # Create new fragment
        new_id = await self.create_fragment(template, fragment.type, fragment.priority)

        # Transfer state
        self.fragments[new_id].state = fragment.state.copy()

        # Update references
        self._update_fragment_references(fragment.id, new_id)

    async def _should_be_active(self, fragment: Fragment) -> bool:
        """Check if fragment should be active."""
        # Check priority
        if fragment.priority in [FragmentPriority.CRITICAL, FragmentPriority.HIGH]:
            return True

        # Check dependencies
        for dep_id in fragment.dependencies:
            if dep_id in self.active_fragments:
                return True

        return False
