import asyncio

import requests
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider

# 1. Output-Schema definieren (Strukturierte Antwort)

class TemperatureResult(BaseModel): 
    station_id: str = Field( description="ID der Wetterstation" ) 
    current_temperature: float = Field( description="Aktuelle Lufttemperatur in Grad Celsius" )
    summary: str = Field( description=( "Kurze Beschreibung der aktuellen Temperatur." "Keine Wettervorhersage erfinden." ) )
  


# 2. Agenten initialisieren
model = OllamaModel(
    model_name="qwen3.6:latest",
    provider=OllamaProvider(base_url="http://localhost:11434/v1"),
)


agent: Agent[None, TemperatureResult] = Agent( model=model, output_type=TemperatureResult, 
                                                system_prompt=("Du bist ein Wetterassistent. " 
                                                               "Wenn der Nutzer nach der aktuellen Temperatur fragt, "
                                                               "verwende das Tool get_temperature. "
                                                               "Gib die Antwort immer im vorgegebenen strukturierten Format zurück. "
                                                               "Die Temperatur wird in Grad Celsius angegeben. " 
                                                               "Erfinde keine Temperaturwerte." ), )




@agent.tool
async def get_temperature(ctx: RunContext[None], station_id: str) -> float:
    """
    Ruft die aktuelle Lufttemperatur einer GeoSphere-Wetterstation ab.

    Args:
        station_id: Die ID der Wetterstation,
                    zum Beispiel 11035 für Wien/Hohe Warte.

    Returns:
        Die aktuelle Lufttemperatur in Grad Celsius.
    """

    url = (
        "https://dataset.api.hub.geosphere.at/"
        "v1/station/current/tawes-v1-10min"
    )

    params = {
        "parameters": "TL",
        "station_ids": station_id,
    }

    # requests is synchronous, so run it in a separate thread
    response = await asyncio.to_thread(
        requests.get,
        url,
        params=params,
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    # Print once to inspect the exact API response
    #print("GeoSphere response:")
    #print(data)

    # Extract the temperature.
    # The exact JSON path may need adjustment depending
    # on the current GeoSphere response format.
    try:
        temperature = (
          data["features"][0]
          ["properties"]
          ["parameters"]
          ["TL"]
          ["data"][0] 
        )

        return float(temperature)

    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(
            "Die Temperatur konnte nicht aus der "
            "GeoSphere-Antwort gelesen werden."
        ) from error


# 5. Run the agent
async def main():
    user_prompt = (
        "Wie hoch ist die aktuelle Temperatur "
        "an der Wetterstation mit der ID 11035?"
    )

    print(f"Frage an den Agenten: '{user_prompt}'\n")

    result = await agent.run(user_prompt)

    # The result is a validated Pydantic object
    current_weather: TemperatureResult = result.output

    print("--- Strukturierte Ausgabe ---")
    print(f"Station:{current_weather.station_id}")
    print(f"Temperatur:"
        f"{current_weather.current_temperature:.1f} °C"
    )
    print(f"{current_weather.summary}")


if __name__ == "__main__":
    asyncio.run(main())