import asyncio
from typing import Any

import requests
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# 1. Output-Schema definieren (Erweitert um Regen und Wind)

class ForecastDay(BaseModel):
    timestamp: str = Field(description="Zeitpunkt der Vorhersage")
    temperature: float = Field(description="Vorhergesagte Temperatur in °C")
    rain: float | None = Field(default=None, description="Niederschlag in mm")
    wind_speed: float | None = Field(default=None, description="Windgeschwindigkeit in km/h")
    wind_direction: float | None = Field(default=None, description="Windrichtung in Grad (0-360°)")


class WeatherResult(BaseModel):
    location: str = Field(description="Name der Stadt oder Wetterstation")

    current_temperature: float | None = Field(
        default=None, description="Aktuell gemessene Temperatur in °C"
    )
    
    current_rain: float | None = Field(
        default=None, description="Aktueller Niederschlag der letzten 10 Min in mm"
    )

    current_wind_speed: float | None = Field(
        default=None, description="Aktuelle Windgeschwindigkeit in km/h"
    )

    forecast: list[ForecastDay] = Field(
        default_factory=list, description="Liste der vorhergesagten Daten"
    )

    summary: str = Field(description="Kurze Zusammenfassung der Wetterdaten inkl. Regen/Wind")

    @field_validator("current_temperature", "current_rain", "current_wind_speed", mode="before")
    @classmethod
    def convert_none_string(cls, value):
        if isinstance(value, str) and value.strip().lower() in {"none", "null", "", "n/a"}:
            return None
        return value


# 2. Agenten initialisieren
model = OllamaModel(
    model_name="qwen3.6:latest",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)

agent: Agent[None, WeatherResult] = Agent(
    model=model,
    output_type=WeatherResult,
    system_prompt=(
        "Du bist ein Wetterassistent für Österreich. "
        "Wenn nach der aktuellen Temperatur/Wetter einer Wetterstation gefragt wird, verwende get_current_weather. "
        "Wenn nach einer Wettervorhersage für eine Stadt gefragt wird, verwende zuerst geocode_city und danach "
        "get_weather_forecast. "
        "Berücksichtige neben der Temperatur auch Regen (Niederschlag) und Wind, sofern vorhanden. "
        "Erfinde keine Daten. "
        "Verwende niemals die Zeichenkette 'None'."
    ),
)


# 3. Custom Tools

@agent.tool
async def get_current_weather(ctx: RunContext[None], station_id: str) -> dict[str, Any]:
    """
    Ruft aktuelle Wetterdaten einer GeoSphere-Station ab.

    Args:
        station_id: Die numerische ID der Wetterstation (z. B. "11035" für Wien/Hohe Warte).
    """
    # Sicherstellen, dass nur Ziffern übergeben wurden
    if not station_id.isdigit():
        raise ValueError(
            f"Ungültige station_id: '{station_id}'. "
            "Es muss eine numerische Stations-ID übergeben werden (z.B. '11035')."
        )

    url = "https://dataset.api.hub.geosphere.at/v1/station/current/tawes-v1-10min"
    params = {
        "parameters": "TL,RR,FF,DD",
        "station_ids": station_id,
    }

    response = await asyncio.to_thread(requests.get, url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    try:
        params_data = data["features"][0]["properties"]["parameters"]
        
        temp = params_data.get("TL", {}).get("data", [None])[0]
        rain = params_data.get("RR", {}).get("data", [None])[0]
        wind_ms = params_data.get("FF", {}).get("data", [None])[0]
        wind_kmh = round(wind_ms * 3.6, 1) if wind_ms is not None else None

        return {
            "temperature": float(temp) if temp is not None else None,
            "rain_mm": float(rain) if rain is not None else None,
            "wind_kmh": wind_kmh,
        }

    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("Aktuelle Wetterdaten konnten nicht gelesen werden.") from error

    
@agent.tool
async def geocode_city(ctx: RunContext[None], city: str) -> dict[str, Any]:
    """Sucht eine Stadt und gibt ihre geografischen Koordinaten zurück."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": f"{city}, Austria", "format": "jsonv2", "limit": 1}
    headers = {"User-Agent": "PydanticAI-Weather_test-App/1.0"}

    response = await asyncio.to_thread(requests.get, url, params=params, headers=headers, timeout=10)

    response.raise_for_status()
    results = response.json()

    if not results:
        raise ValueError(f"Keine Koordinaten für '{city}' gefunden.")

    result = results[0]
    return {
        "city": result["display_name"],
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
    }


@agent.tool
async def get_weather_forecast(ctx: RunContext[None], latitude: float, longitude: float) -> list[dict[str, Any]]:
    """ Ruft Temperatur, Niederschlag (rr) und Wind (ff, dd) Vorhersage von GeoSphere Austria ab.

    Args:
        ctx (RunContext[None]): Context
        latitude (float): latitude der Stadt
        longitude (float): longitude der Stadt

    Returns:
        list[dict[str, Any]]: Vorhersagewerte inkl. Temperatur, Regen und Wind
    """    

    url = "https://dataset.api.hub.geosphere.at/v1/timeseries/forecast/nwp-v1-1h-2500m"

    params = {
        "parameters": "t2m",
        "lat_lon": f"{latitude},{longitude}",
    }

    response = await asyncio.to_thread(requests.get, url, params=params, timeout=20)
    if response.status_code >= 400:
        raise RuntimeError(f"GeoSphere forecast request failed: {response.status_code} {response.text}")

    data = response.json()

    try:
        timestamps = data["timestamps"]
        params_data = data["features"][0]["properties"]["parameters"]

        temps = params_data.get("t2m", {}).get("data", [])

        forecast = []
        for ts, t in zip(timestamps, temps):
            if t is not None:
                forecast.append(
                    {
                        "timestamp": ts,
                        "temperature": float(t),
                        "rain": None,
                        "wind_speed": None,
                        "wind_direction": None,
                    }
                )

        return forecast

    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError("Vorhersagedaten konnten nicht gelesen werden.") from error

# 4. Main-Funktion

async def main(city: str):
    
    user_prompt = (
        f"Wie ist das Wetter (Temperatur, Regen, Wind) in {city} im Moment und wie "
        f"sieht die Vorhersage aus?"
    )

    print(f"Frage an den Agenten:\n{user_prompt}\n")

    result = await agent.run(user_prompt)
    weather: WeatherResult = result.output

    print("--- Wetterausgabe ---")
    print(f"Ort: {weather.location}")

    if weather.current_temperature is not None:
        print(f"Aktuelle Temperatur:  {weather.current_temperature:.1f} °C")
    if weather.current_rain is not None:
        print(f"Aktueller Niederschlag: {weather.current_rain:.1f} mm")
    if weather.current_wind_speed is not None:
        print(f"Aktueller Wind:        {weather.current_wind_speed:.1f} km/h")

    if weather.forecast:
        print("\nVorhersage:")
      
        for item in weather.forecast:
            rain_str = f", Regen: {item.rain:.1f}mm" if item.rain else ""
            wind_str = f", Wind: {item.wind_speed:.1f}km/h" if item.wind_speed else ""
            print(f"  {item.timestamp}: {item.temperature:.1f} °C{rain_str}{wind_str}")

    print(f"\nZusammenfassung: {weather.summary}")


if __name__ == "__main__":
    asyncio.run(main("Wien"))