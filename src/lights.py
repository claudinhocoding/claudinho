"""
Smart Home — Kasa Device Control
=================================

Controls TP-Link Kasa/Tapo smart devices over local network.
Uses python-kasa for direct LAN communication (no cloud needed).

Devices are discovered on startup and cached.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Lazy imports — python-kasa might not be installed
_kasa_available = False
try:
    from kasa import Discover, Device
    _kasa_available = True
except ImportError:
    pass


class SmartHome:
    """Manages Kasa smart home devices."""

    def __init__(self):
        self.devices: Dict[str, Device] = {}  # name -> device
        self._discovered = False

    async def _discover(self) -> Dict[str, Device]:
        """Discover all Kasa devices on the local network."""
        username = os.environ.get("KASA_USERNAME", "")
        password = os.environ.get("KASA_PASSWORD", "")

        kwargs = {}
        if username and password:
            from kasa import Credentials
            kwargs["credentials"] = Credentials(username, password)

        logger.info("🏠 Discovering smart home devices...")
        try:
            devices = await Discover.discover(**kwargs)
        except Exception as e:
            logger.error(f"Device discovery failed: {e}")
            return {}

        result = {}
        for ip, dev in devices.items():
            try:
                await dev.update()
                name = dev.alias or f"device_{ip}"
                result[name.lower()] = dev
                logger.info(f"  📍 {name} ({dev.model}) @ {ip} — {'ON' if dev.is_on else 'OFF'}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to query {ip}: {e}")

        return result

    def discover(self):
        """Synchronous wrapper for device discovery."""
        if not _kasa_available:
            logger.warning("python-kasa not installed (pip install python-kasa)")
            return

        try:
            self.devices = asyncio.get_event_loop().run_until_complete(self._discover())
        except RuntimeError:
            # No event loop running
            self.devices = asyncio.run(self._discover())

        self._discovered = True
        logger.info(f"🏠 Found {len(self.devices)} device(s)")

    def get_device_list(self) -> List[str]:
        """Get list of device descriptions for the system prompt."""
        descriptions = []
        for name, dev in self.devices.items():
            status = "on" if dev.is_on else "off"
            device_type = "light" if hasattr(dev, 'brightness') else "plug"
            desc = f"- {dev.alias} ({device_type}, currently {status})"
            if hasattr(dev, 'brightness') and dev.is_on:
                desc += f", brightness: {dev.brightness}%"
            descriptions.append(desc)
        return descriptions

    def _find_device(self, name: str) -> Optional[Device]:
        """Find a device by name (case-insensitive, partial match)."""
        name_lower = name.lower().strip()
        # Exact match
        if name_lower in self.devices:
            return self.devices[name_lower]
        # Partial match
        for dev_name, dev in self.devices.items():
            if name_lower in dev_name or dev_name in name_lower:
                return dev
        return None

    async def _turn_on(self, device: Device):
        await device.turn_on()
        await device.update()

    async def _turn_off(self, device: Device):
        await device.turn_off()
        await device.update()

    async def _set_brightness(self, device: Device, level: int):
        if hasattr(device, 'set_brightness'):
            await device.set_brightness(level)
            await device.update()
        else:
            logger.warning(f"{device.alias} doesn't support brightness")

    def _run_async(self, coro):
        """Run an async coroutine from sync context."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.ensure_future(coro)
            return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def turn_on(self, device_name: str) -> str:
        """Turn on a device. Returns status message."""
        dev = self._find_device(device_name)
        if not dev:
            return f"Device '{device_name}' not found"
        self._run_async(self._turn_on(dev))
        logger.info(f"💡 Turned ON: {dev.alias}")
        return f"Turned on {dev.alias}"

    def turn_off(self, device_name: str) -> str:
        """Turn off a device. Returns status message."""
        dev = self._find_device(device_name)
        if not dev:
            return f"Device '{device_name}' not found"
        self._run_async(self._turn_off(dev))
        logger.info(f"🌑 Turned OFF: {dev.alias}")
        return f"Turned off {dev.alias}"

    def set_brightness(self, device_name: str, level: int) -> str:
        """Set brightness (0-100). Returns status message."""
        dev = self._find_device(device_name)
        if not dev:
            return f"Device '{device_name}' not found"
        level = max(0, min(100, level))
        self._run_async(self._set_brightness(dev, level))
        logger.info(f"🔆 Brightness {dev.alias}: {level}%")
        return f"Set {dev.alias} brightness to {level}%"

    def execute_action(self, action: str) -> str:
        """
        Execute a structured action string.
        
        Formats:
            <<turn_on:Device Name>>
            <<turn_off:Device Name>>
            <<brightness:Device Name:50>>
            <<toggle:Device Name>>
        """
        action = action.strip()
        parts = action.split(":")
        if len(parts) < 2:
            return f"Invalid action: {action}"

        command = parts[0].strip().lower()
        device_name = parts[1].strip()

        if command == "turn_on":
            return self.turn_on(device_name)
        elif command == "turn_off":
            return self.turn_off(device_name)
        elif command == "brightness" and len(parts) >= 3:
            try:
                level = int(parts[2].strip())
                return self.set_brightness(device_name, level)
            except ValueError:
                return f"Invalid brightness: {parts[2]}"
        elif command == "toggle":
            dev = self._find_device(device_name)
            if dev:
                if dev.is_on:
                    return self.turn_off(device_name)
                else:
                    return self.turn_on(device_name)
            return f"Device '{device_name}' not found"
        else:
            return f"Unknown command: {command}"
