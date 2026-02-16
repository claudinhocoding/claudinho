"""
Spotify Music Control
=====================

Controls Spotify playback via the Web API using spotipy.
Requires spotifyd running on the Pi for audio output.

Setup:
  1. pip install spotipy
  2. Run scripts/spotify_auth.py once to authorize
  3. spotifyd running as systemd service
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_sp = None


def _get_client():
    """Get or create the Spotify client (lazy init)."""
    global _sp
    if _sp is not None:
        return _sp

    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError:
        logger.warning("spotipy not installed (pip install spotipy)")
        return None

    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
    redirect_uri = os.environ.get("SPOTIFY_REDIRECT_URI", "https://localhost:8888/callback")

    if not client_id or not client_secret:
        logger.warning("Spotify credentials not set (SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)")
        return None

    cache_path = os.path.expanduser("~/.claudinho_spotify_cache")

    try:
        auth_manager = SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope="user-modify-playback-state user-read-playback-state user-read-currently-playing",
            cache_path=cache_path,
            open_browser=False,
        )
        _sp = spotipy.Spotify(auth_manager=auth_manager)
        # Quick test
        _sp.current_user()
        logger.info("🎵 Spotify connected")
        return _sp
    except Exception as e:
        logger.warning(f"Spotify auth failed: {e}")
        logger.warning("Run: python3 scripts/spotify_auth.py")
        return None


def _get_device_id() -> Optional[str]:
    """Find the Pi's spotifyd device."""
    sp = _get_client()
    if not sp:
        return None
    try:
        devices = sp.devices()
        for dev in devices.get("devices", []):
            # Prefer the Pi's spotifyd instance
            name = dev.get("name", "").lower()
            if "claudinho" in name or "spotifyd" in name or "raspberrypi" in name:
                return dev["id"]
        # Fall back to any active device
        for dev in devices.get("devices", []):
            if dev.get("is_active"):
                return dev["id"]
        # Fall back to first device
        if devices.get("devices"):
            return devices["devices"][0]["id"]
    except Exception as e:
        logger.error(f"Failed to get devices: {e}")
    return None


class MusicPlayer:
    """Spotify music controller."""

    def __init__(self):
        self.available = _get_client() is not None

    def get_status(self) -> str:
        """Get current playback description for system prompt."""
        sp = _get_client()
        if not sp:
            return "Spotify not connected"
        try:
            current = sp.current_playback()
            if not current or not current.get("is_playing"):
                return "Nothing playing"
            track = current.get("item", {})
            name = track.get("name", "Unknown")
            artists = ", ".join(a["name"] for a in track.get("artists", []))
            return f"Now playing: {name} by {artists}"
        except Exception:
            return "Spotify status unavailable"

    def play(self, query: str) -> str:
        """Search and play a song, artist, album, or playlist."""
        sp = _get_client()
        if not sp:
            return "Spotify not connected"

        device_id = _get_device_id()
        if not device_id:
            return "No Spotify device found. Is spotifyd running?"

        try:
            # Search for the query
            results = sp.search(q=query, limit=1, type="track,playlist,album,artist")

            # Try tracks first
            tracks = results.get("tracks", {}).get("items", [])
            if tracks:
                track = tracks[0]
                name = track["name"]
                artist = track["artists"][0]["name"]
                sp.start_playback(device_id=device_id, uris=[track["uri"]])
                logger.info(f"🎵 Playing: {name} by {artist}")
                return f"Playing {name} by {artist}"

            # Try playlists
            playlists = results.get("playlists", {}).get("items", [])
            if playlists:
                playlist = playlists[0]
                sp.start_playback(device_id=device_id, context_uri=playlist["uri"])
                logger.info(f"🎵 Playing playlist: {playlist['name']}")
                return f"Playing playlist {playlist['name']}"

            # Try albums
            albums = results.get("albums", {}).get("items", [])
            if albums:
                album = albums[0]
                sp.start_playback(device_id=device_id, context_uri=album["uri"])
                logger.info(f"🎵 Playing album: {album['name']}")
                return f"Playing album {album['name']}"

            # Try artists (play top tracks)
            artists = results.get("artists", {}).get("items", [])
            if artists:
                artist = artists[0]
                sp.start_playback(device_id=device_id, context_uri=artist["uri"])
                logger.info(f"🎵 Playing artist: {artist['name']}")
                return f"Playing {artist['name']}"

            return f"Couldn't find anything for '{query}'"

        except Exception as e:
            logger.error(f"Spotify play error: {e}")
            return f"Failed to play: {e}"

    def pause(self) -> str:
        """Pause playback."""
        sp = _get_client()
        if not sp:
            return "Spotify not connected"
        try:
            sp.pause_playback()
            logger.info("⏸️  Paused")
            return "Paused"
        except Exception as e:
            return f"Failed to pause: {e}"

    def resume(self) -> str:
        """Resume playback."""
        sp = _get_client()
        if not sp:
            return "Spotify not connected"
        device_id = _get_device_id()
        try:
            sp.start_playback(device_id=device_id)
            logger.info("▶️  Resumed")
            return "Resumed"
        except Exception as e:
            return f"Failed to resume: {e}"

    def skip(self) -> str:
        """Skip to next track."""
        sp = _get_client()
        if not sp:
            return "Spotify not connected"
        try:
            sp.next_track()
            logger.info("⏭️  Skipped")
            return "Skipped to next track"
        except Exception as e:
            return f"Failed to skip: {e}"

    def previous(self) -> str:
        """Go to previous track."""
        sp = _get_client()
        if not sp:
            return "Spotify not connected"
        try:
            sp.previous_track()
            logger.info("⏮️  Previous")
            return "Previous track"
        except Exception as e:
            return f"Failed: {e}"

    def set_volume(self, level: int) -> str:
        """Set volume (0-100)."""
        sp = _get_client()
        if not sp:
            return "Spotify not connected"
        level = max(0, min(100, level))
        try:
            sp.volume(level)
            logger.info(f"🔊 Volume: {level}%")
            return f"Volume set to {level}%"
        except Exception as e:
            return f"Failed to set volume: {e}"

    def queue(self, query: str) -> str:
        """Add a track to the queue."""
        sp = _get_client()
        if not sp:
            return "Spotify not connected"
        try:
            results = sp.search(q=query, limit=1, type="track")
            tracks = results.get("tracks", {}).get("items", [])
            if tracks:
                track = tracks[0]
                sp.add_to_queue(track["uri"])
                name = track["name"]
                artist = track["artists"][0]["name"]
                logger.info(f"📋 Queued: {name} by {artist}")
                return f"Added {name} by {artist} to queue"
            return f"Couldn't find '{query}'"
        except Exception as e:
            return f"Failed to queue: {e}"

    def execute_action(self, action: str) -> str:
        """Execute a structured action string."""
        parts = action.split(":", 1)
        command = parts[0].strip().lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if command == "spotify_play":
            return self.play(arg)
        elif command == "spotify_pause":
            return self.pause()
        elif command == "spotify_resume":
            return self.resume()
        elif command == "spotify_skip":
            return self.skip()
        elif command == "spotify_previous":
            return self.previous()
        elif command == "spotify_volume":
            try:
                return self.set_volume(int(arg))
            except ValueError:
                return f"Invalid volume: {arg}"
        elif command == "spotify_queue":
            return self.queue(arg)
        else:
            return f"Unknown music command: {command}"
