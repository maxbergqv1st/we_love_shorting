# UI Facelift — plan

Branch: `feature/ui-facelift`

## Mål

Göra `app.py` tydligare och snyggare utan att röra någon backend-logik
(`src/we_love_shorting/`). Just nu är det otydligt i UI:t **vad som tränas**,
**vilken modell** som används och **vad target/features är** — den här planen
löser det.

## Vad som ändras

**Ny fil:**
- `.streamlit/config.toml` — färgtema (ingen Python-logik, bara config)

**Ändrad fil:**
- `app.py` — enda Python-filen som rörs

## De 5 delarna

1. **Tema** — konsekvent färgpalett via `config.toml` istället för Streamlits
   default-utseende.
2. **Statusbanner** — en tydlig ruta som säger t.ex. "Du tränar: Linear
   Regression → förutsäger **Guld** från Nyhetston + S&P 500", byggd
   dynamiskt från redan valda target/features.
3. **Sidebar** — flytta inställningar (sökord, fyll-knappar, target/features)
   till sidopanelen. Huvudytan visar bara resultat (graf, tabell, banner).
4. **Snabb-metrics** — `st.metric`-kort ovanför grafen (senaste faktiska
   värde, senaste prediktion, differens).
5. **Modellväljare (skuggad)** — en väljare med Linear Regression aktiv och
   framtida alternativ (Random Forest, Ridge) synliga men markerade
   "kommer snart" / inaktiverade. Förbereder UI:t för nästa persons jobb
   (modelljämförelse) utan att bygga den logiken nu.

## Vad som INTE rörs

- `src/we_love_shorting/controller.py`, `db.py`, `features.py`,
  `signal_model.py`, `sources/` — all backend-logik lämnas orörd
- `tests/` — inga nya tester behövs (ren UI/config, ingen ny logik)

## Process

1. Implementera i den här branchen, en del i taget
2. Testa lokalt i browsern efter varje del
3. `ruff check --fix .` + `ruff format .` innan commit
4. Manuella commits, en per logisk del
5. Visa branchen för gruppen innan PR
6. Push + PR mot `main` efter godkännande
