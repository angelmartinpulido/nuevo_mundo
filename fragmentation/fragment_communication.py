"""
Communication system for fragments within and across nodes.
"""

import asyncio
import hashlib
from typing import Dict, List, Set, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum
import aiofiles
import json
from cryptography.fernet import Fernet
from .fragment_manager import Fragment, FragmentPriority, FragmentType


class MessageType(Enum):
    SYNC = 0
    STATE = 1
    COMMAND = 2
    RESPONSE = 3
    HEALTH = 4


@dataclass
class FragmentMessage:
    id: str
    type: MessageType
    sender: str
    receiver: str
    data: Any
    timestamp: float
    signature: bytes


class FragmentCommunication:
    def __init__(self):
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.active_channels: Dict[str, Set[str]] = {}
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.message_handlers = {
            MessageType.SYNC: self._handle_sync,
            MessageType.STATE: self._handle_state,
            MessageType.COMMAND: self._handle_command,
            MessageType.RESPONSE: self._handle_response,
            MessageType.HEALTH: self._handle_health,
        }

    async def initialize(self):
        """Initialize communication system."""
        await self._setup_queues()
        await self._setup_channels()
        await self._start_message_processor()

    async def send_message(
        self, sender: str, receiver: str, msg_type: MessageType, data: Any
    ) -> bool:
        """Send message between fragments."""
        try:
            # Create message
            message = await self._create_message(sender, receiver, msg_type, data)

            # Get appropriate queue
            queue = await self._get_message_queue(receiver)

            # Send message
            await queue.put(message)

            return True

        except Exception as e:
            logging.error(f"Failed to send message: {e}")
            return False

    async def _create_message(
        self, sender: str, receiver: str, msg_type: MessageType, data: Any
    ) -> FragmentMessage:
        """Create a new message."""
        # Serialize data
        serialized_data = json.dumps(data)

        # Encrypt data
        encrypted_data = self.fernet.encrypt(serialized_data.encode())

        # Create message
        message = FragmentMessage(
            id=self._generate_message_id(),
            type=msg_type,
            sender=sender,
            receiver=receiver,
            data=encrypted_data,
            timestamp=asyncio.get_event_loop().time(),
            signature=await self._sign_message(encrypted_data),
        )

        return message

    async def _process_messages(self):
        """Process incoming messages."""
        while True:
            # Process messages for each fragment
            for fragment_id, queue in self.message_queues.items():
                try:
                    # Get message if available
                    if not queue.empty():
                        message = await queue.get()

                        # Verify message
                        if await self._verify_message(message):
                            # Handle message
                            await self._handle_message(message)

                except Exception as e:
                    logging.error(f"Message processing error: {e}")

            await asyncio.sleep(0.01)  # Small delay to prevent CPU overuse

    async def _handle_message(self, message: FragmentMessage):
        """Handle incoming message."""
        handler = self.message_handlers.get(message.type)
        if handler:
            await handler(message)

    async def _handle_sync(self, message: FragmentMessage):
        """Handle synchronization message."""
        try:
            # Decrypt data
            decrypted_data = self._decrypt_message_data(message)

            # Update fragment state
            await self._update_fragment_state(message.receiver, decrypted_data)

            # Send acknowledgment
            await self.send_message(
                message.receiver,
                message.sender,
                MessageType.RESPONSE,
                {"status": "sync_complete"},
            )

        except Exception as e:
            logging.error(f"Sync handling error: {e}")

    async def _handle_state(self, message: FragmentMessage):
        """Handle state update message."""
        try:
            # Decrypt state data
            state_data = self._decrypt_message_data(message)

            # Verify state integrity
            if await self._verify_state_integrity(state_data):
                # Update fragment state
                await self._update_fragment_state(message.receiver, state_data)

                # Propagate state update if needed
                await self._propagate_state_update(message)

        except Exception as e:
            logging.error(f"State handling error: {e}")

    async def _handle_command(self, message: FragmentMessage):
        """Handle command message."""
        try:
            # Decrypt command data
            command_data = self._decrypt_message_data(message)

            # Verify command authorization
            if await self._verify_command_authorization(message):
                # Execute command
                result = await self._execute_command(message.receiver, command_data)

                # Send response
                await self.send_message(
                    message.receiver,
                    message.sender,
                    MessageType.RESPONSE,
                    {"status": "success", "result": result},
                )

        except Exception as e:
            logging.error(f"Command handling error: {e}")

    async def _handle_response(self, message: FragmentMessage):
        """Handle response message."""
        try:
            # Decrypt response data
            response_data = self._decrypt_message_data(message)

            # Update response tracking
            await self._update_response_tracking(message.sender, response_data)

            # Handle response callbacks
            await self._handle_response_callbacks(message)

        except Exception as e:
            logging.error(f"Response handling error: {e}")

    async def _handle_health(self, message: FragmentMessage):
        """Handle health check message."""
        try:
            # Decrypt health data
            health_data = self._decrypt_message_data(message)

            # Update health status
            await self._update_health_status(message.sender, health_data)

            # Send health response
            await self._send_health_response(message)

        except Exception as e:
            logging.error(f"Health handling error: {e}")

    async def establish_channel(self, fragment1: str, fragment2: str) -> bool:
        """Establish communication channel between fragments."""
        try:
            # Create channel
            channel_id = self._generate_channel_id(fragment1, fragment2)

            # Setup channel encryption
            channel_key = await self._setup_channel_encryption(channel_id)

            # Register channel
            self.active_channels.setdefault(fragment1, set()).add(fragment2)
            self.active_channels.setdefault(fragment2, set()).add(fragment1)

            return True

        except Exception as e:
            logging.error(f"Channel establishment failed: {e}")
            return False

    async def close_channel(self, fragment1: str, fragment2: str):
        """Close communication channel between fragments."""
        try:
            # Remove channel registrations
            if fragment1 in self.active_channels:
                self.active_channels[fragment1].discard(fragment2)
            if fragment2 in self.active_channels:
                self.active_channels[fragment2].discard(fragment1)

            # Cleanup channel resources
            channel_id = self._generate_channel_id(fragment1, fragment2)
            await self._cleanup_channel(channel_id)

        except Exception as e:
            logging.error(f"Channel closure failed: {e}")

    async def broadcast_message(
        self, sender: str, msg_type: MessageType, data: Any
    ) -> bool:
        """Broadcast message to all connected fragments."""
        try:
            # Get connected fragments
            connected_fragments = self.active_channels.get(sender, set())

            # Send to all connected fragments
            results = await asyncio.gather(
                *[
                    self.send_message(sender, receiver, msg_type, data)
                    for receiver in connected_fragments
                ]
            )

            return all(results)

        except Exception as e:
            logging.error(f"Broadcast failed: {e}")
            return False

    async def sync_fragment_state(
        self, fragment_id: str, target_fragments: List[str]
    ) -> bool:
        """Synchronize fragment state with target fragments."""
        try:
            # Get fragment state
            state = await self._get_fragment_state(fragment_id)

            # Send sync messages
            results = await asyncio.gather(
                *[
                    self.send_message(fragment_id, target, MessageType.SYNC, state)
                    for target in target_fragments
                ]
            )

            return all(results)

        except Exception as e:
            logging.error(f"State sync failed: {e}")
            return False

    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

    def _generate_channel_id(self, fragment1: str, fragment2: str) -> str:
        """Generate unique channel ID."""
        return hashlib.sha256(f"{fragment1}:{fragment2}".encode()).hexdigest()[:16]

    async def _verify_message(self, message: FragmentMessage) -> bool:
        """Verify message integrity and authenticity."""
        try:
            # Verify signature
            if not await self.security_manager.verify_signature(
                message.data, message.signature
            ):
                return False

            # Verify timestamp
            if not self._verify_timestamp(message.timestamp):
                return False

            # Verify sender/receiver
            if not await self._verify_fragments(message.sender, message.receiver):
                return False

            return True

        except Exception:
            return False

    def _verify_timestamp(self, timestamp: float) -> bool:
        """Verify message timestamp is recent."""
        current_time = asyncio.get_event_loop().time()
        return abs(current_time - timestamp) < 300  # 5 minutes max difference

    async def _verify_fragments(self, sender: str, receiver: str) -> bool:
        """Verify both fragments exist and are active."""
        return (
            sender in self.fragments
            and receiver in self.fragments
            and sender in self.active_fragments
            and receiver in self.active_fragments
        )

    def _decrypt_message_data(self, message: FragmentMessage) -> Any:
        """Decrypt message data."""
        decrypted = self.fernet.decrypt(message.data)
        return json.loads(decrypted.decode())

    async def _update_fragment_state(self, fragment_id: str, state_data: Dict):
        """Update fragment state."""
        fragment = self.fragments.get(fragment_id)
        if fragment:
            fragment.state.update(state_data)

    async def _verify_command_authorization(self, message: FragmentMessage) -> bool:
        """Verify if sender is authorized to send commands to receiver."""
        sender_fragment = self.fragments.get(message.sender)
        receiver_fragment = self.fragments.get(message.receiver)

        if not sender_fragment or not receiver_fragment:
            return False

        # Check if sender has required priority
        return sender_fragment.priority.value <= receiver_fragment.priority.value

    async def _execute_command(self, fragment_id: str, command_data: Dict) -> Any:
        """Execute command on fragment."""
        fragment = self.fragments.get(fragment_id)
        if not fragment:
            raise ValueError("Fragment not found")

        # Execute command in isolated environment
        return await self._execute_in_isolation(fragment, command_data)

    async def _execute_in_isolation(
        self, fragment: Fragment, command_data: Dict
    ) -> Any:
        """Execute command in isolated environment."""
        # Create isolated namespace
        namespace = {"fragment": fragment, "command": command_data, "result": None}

        # Execute in isolation
        exec(command_data["code"], namespace)

        return namespace["result"]

    async def _propagate_state_update(self, message: FragmentMessage):
        """Propagate state update to dependent fragments."""
        fragment = self.fragments.get(message.receiver)
        if not fragment:
            return

        # Get dependent fragments
        dependents = self._get_dependent_fragments(message.receiver)

        # Propagate update
        for dependent in dependents:
            await self.send_message(
                message.receiver,
                dependent,
                MessageType.STATE,
                self._decrypt_message_data(message),
            )

    def _get_dependent_fragments(self, fragment_id: str) -> Set[str]:
        """Get fragments that depend on given fragment."""
        dependents = set()

        for fid, fragment in self.fragments.items():
            if fragment_id in fragment.dependencies:
                dependents.add(fid)

        return dependents
