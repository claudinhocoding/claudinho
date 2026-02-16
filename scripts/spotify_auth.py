#!/usr/bin/env python3
"""
One-time Spotify OAuth authorization.

Run this once on the Pi to authorize Claudinho to control your Spotify.
The token is cached and auto-refreshes — you only need to do this once.

Usage:
  cd ~/claudinho
  source venv/bin/activate
  python3 scripts/spotify_auth.py
"""

import os
import sys

# Load .env
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except ImportError:
    print("Install spotipy first: pip install spotipy")
    sys.exit(1)

client_id = os.environ.get("SPOTIFY_CLIENT_ID", "")
client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
redirect_uri = os.environ.get("SPOTIFY_REDIRECT_URI", "https://localhost:8888/callback")

if not client_id or not client_secret:
    print("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env")
    sys.exit(1)

cache_path = os.path.expanduser("~/.claudinho_spotify_cache")

print("=== Spotify Authorization for Claudinho ===\n")
print(f"Client ID: {client_id[:8]}...")
print(f"Redirect URI: {redirect_uri}")
print(f"Token cache: {cache_path}\n")

auth_manager = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope="user-modify-playback-state user-read-playback-state user-read-currently-playing",
    cache_path=cache_path,
    open_browser=False,
)

# Check if we already have a valid token
token = auth_manager.get_cached_token()
if token:
    print("✅ Already authorized! Token is cached and valid.")
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user = sp.current_user()
    print(f"   Logged in as: {user['display_name']} ({user['id']})")
    sys.exit(0)

# Need to authorize
auth_url = auth_manager.get_authorize_url()
print("1. Open this URL in any browser (Mac, phone, etc.):\n")
print(f"   {auth_url}\n")
print("2. Log into Spotify and authorize the app.")
print("3. You'll be redirected to a URL that starts with your redirect URI.")
print("   (The page won't load — that's fine!)")
print("4. Copy the FULL URL from your browser's address bar and paste it here:\n")

response_url = input("Paste URL here: ").strip()

try:
    code = auth_manager.parse_response_code(response_url)
    token = auth_manager.get_access_token(code)
    
    sp = spotipy.Spotify(auth_manager=auth_manager)
    user = sp.current_user()
    
    print(f"\n✅ Success! Logged in as: {user['display_name']} ({user['id']})")
    print(f"   Token cached at: {cache_path}")
    print("\n   You can now restart Claudinho — Spotify is ready!")
except Exception as e:
    print(f"\n❌ Authorization failed: {e}")
    print("   Make sure you copied the complete URL.")
    sys.exit(1)
