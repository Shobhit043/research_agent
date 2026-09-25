"""Time and place: current local time anywhere, and basic facts about places.

Place lookup uses the Open-Meteo geocoding API (GeoNames data, no API key). Local times come
from the IANA timezone database via `zoneinfo`, never from the model's guess of the date.
"""

import datetime
import json
from urllib.parse import urlencode
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import BaseTool, tool

from agent.config import Settings
from agent.tools.http import cache, safe_get

GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
_MAX_BYTES = 200_000


def geocode(query: str, settings: Settings, count: int = 3) -> list[dict]:
    """Best-matching places for a name, largest/most relevant first."""
    params = {"name": query, "count": str(count), "language": "en", "format": "json"}
    url = f"{GEOCODING_API}?{urlencode(params)}"
    body = cache.get(("geo", url))
    if body is None:
        body = safe_get(url, settings.request_timeout, _MAX_BYTES).body
        cache.set(("geo", url), body)
    return json.loads(body).get("results", [])


def describe_time(zone: ZoneInfo, now: datetime.datetime | None = None) -> str:
    local = (now or datetime.datetime.now(datetime.UTC)).astimezone(zone)
    offset = local.strftime("%z")
    return f"{local:%A %d %B %Y, %H:%M} (UTC{offset[:3]}:{offset[3:]}, {zone.key})"


def place_label(place: dict) -> str:
    parts = [place.get("name"), place.get("admin1"), place.get("country")]
    return ", ".join(dict.fromkeys(p for p in parts if p))


def build_get_time_tool(settings: Settings) -> BaseTool:
    @tool
    def get_time(location: str = "UTC") -> str:
        """
        Get the current date and time in a city, region or country, or in an IANA
        timezone such as "Europe/Paris" or "UTC". Use this whenever the answer depends
        on today's date or the time somewhere, instead of assuming it.

        Args:
            location: A place name (e.g. "Tokyo", "São Paulo") or an IANA timezone.
        """
        now = datetime.datetime.now(datetime.UTC)
        try:
            return f"Current time in {location}: {describe_time(ZoneInfo(location), now)}"
        except (ZoneInfoNotFoundError, ValueError, OSError):
            pass  # not a timezone name; treat it as a place
        places = geocode(location, settings, count=1)
        if not places or not places[0].get("timezone"):
            return f"Could not find a place or timezone called '{location}'."
        place = places[0]
        return f"Current time in {place_label(place)}: {describe_time(ZoneInfo(place['timezone']), now)}"

    return get_time


def build_place_info_tool(settings: Settings) -> BaseTool:
    @tool
    def place_info(query: str) -> str:
        """
        Look up a place (city, town, region or country) and return its country and
        region, coordinates, elevation, population, timezone and current local time.
        Several matches are returned when the name is ambiguous (e.g. "Springfield").

        Args:
            query: The place name, optionally with a country, e.g. "Paris, Texas".
        """
        name, _, hint = query.partition(",")
        places = geocode(name.strip(), settings, count=5)
        hint = hint.strip().lower()
        if hint:
            # "Paris, Texas": prefer matches whose region or country contains the hint.
            places.sort(key=lambda p: hint not in f"{p.get('admin1', '')} {p.get('country', '')}".lower())
        if not places:
            return f"No place found matching '{query}'."

        sections = []
        for number, place in enumerate(places[:3], start=1):
            lat, lon = place["latitude"], place["longitude"]
            facts = [f"Coordinates: {lat:.4f}, {lon:.4f}"]
            if place.get("elevation") is not None:
                facts.append(f"Elevation: {place['elevation']:.0f} m")
            if place.get("population"):
                facts.append(f"Population: {place['population']:,}")
            if place.get("timezone"):
                facts.append(f"Timezone: {place['timezone']}")
                facts.append(f"Local time now: {describe_time(ZoneInfo(place['timezone']))}")
            url = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=11/{lat}/{lon}"
            sections.append(f"[P{number}] {place_label(place)}\nURL: {url}\n" + "\n".join(facts))
        return "\n\n".join(sections)

    return place_info
