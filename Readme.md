# Pydanic AI Testing - Automate Weather Reports
This Tool extracts the Temperature, Precipitation and Wind from a given City from Geosphere Austria API. The Name of a given city is matched to a station ID. Current temperature and forecasts are printed out and a verbal weather forcast as well. 
To format outputs Pydantic AI is used.

As LLM for output generation a local Ollama installation is used. 

Example:

`Wien`

Ausgabe


**--- Wetterausgabe ---**<br>
Ort: Wien<br>
Aktuelle Temperatur:  33.5 °C <br>
Aktueller Niederschlag: 0.0 mm <br>
Aktueller Wind:        17.3 km/h <br>

**Vorhersage:<br>**
  2026-08-06T15:00+00:00: 35.4 °C<br>
  2026-08-06T16:00+00:00: 34.3 °C<br>
  2026-08-06T17:00+00:00: 29.0 °C<br>
  2026-08-06T18:00+00:00: 29.4 °C<br>
  2026-08-06T19:00+00:00: 28.8 °C<br>
  2026-08-06T20:00+00:00: 26.2 °C<br>
  2026-08-07T12:00+00:00: 26.7 °C<br>
  2026-08-07T15:00+00:00: 27.5 °C<br>
  2026-08-07T18:00+00:00: 25.4 °C<br>
  2026-08-08T12:00+00:00: 28.1 °C<br>

**Zusammenfassung:** In Wien herrschen aktuell warme 33,5°C bei leichter Brise (17,3 km/h) und völlig trockener Luft. Der Tag wird mit Höchstwerten um die 35,4°C am frühen Nachmittag sehr warm, kühlt dann aber allmählich auf etwa 20–21°C in der Nacht ab. In der Vorschau sind keine Niederschläge zu erwarten.


--------------------------
**Requierements**

A running Ollama instance and qwen3.6:latest model installed