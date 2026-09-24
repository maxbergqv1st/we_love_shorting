# FLOW — exakt hur allt rör sig

Referens för hur data och kontroll flödar genom appen, steg för steg. Läs
`docs/PRESENTATION.md` för *varför*; det här är *hur*.

## 0. Grundmodellen: Streamlit kör om allt, hela tiden

Varje widget-interaktion kör `app.py` uppifrån och ner igen. Allt som är dyrt
ligger därför bakom en cache, och allt tillstånd som ska överleva en omkörning
ligger i `st.session_state`. Att förstå appen = att veta vad som händer i en
omkörning och vad som är cachetradfat.

## 1. Kallstart (första körningen i en session)

1. `app.py` importeras: `LABELS` byggs (en etikett per kolumn — suffixmängden
   ägs av `features.STREAM_SUFFIXES`, formaten av `_SUFFIX_FORMATS`;
   `register_stream_labels` är enda skrivvägen), cache-wrappers definieras.
2. `st.session_state["df"]` saknas → info-ruta + `st.stop()`. Inget mer körs
   förrän användaren fyllt och laddat data via **Setup**.

## 2. Datavägen (Setup-dialogen — enda vägen in i DB:n)

```
"Första fyllning"/"Fyll på till idag"
  → controller.backfill()/update()
      → sources/gdelt.fetch_tone()   (urllib; @retry på HTTP 429)
      → sources/yahoo.fetch_prices() (yfinance, en per ticker)
      → db.upsert(tabell, frame)     (whitelistad tabell, idempotent;
                                      varje källa committar oberoende)
  → get_data.clear()                 (DB ändrad → cachen ogiltig)

"Ladda data"
  → get_data()  = st.cache_data(1h) runt controller.get_data
      → db.load() per tabell         (läser BARA DB:n — inget nätverk)
      → features.build_features():
           per ström, på strömmens EGEN kalender: ret/ma5/ma21/vol21
           → outer-merge på datum → ffill (carry-forward över stängda dagar)
           → merge_asof mot tone (dagliga rader) → dropna
           → market_closed-flaggan (referenskalender = S&P 500)
  → st.session_state["df"] = den breda tabellen  → st.rerun()
```

`df` i session_state är sedan **rådataframen** — en rad per tone-datum, alla
kolumner för alla strömmar. Den ändras bara av Setup eller stream-vävningen.

## 3. En vanlig omkörning, uppifrån och ner

1. **Topbar** ritas; datumspann/radantal fylls i EFTER steg 3 (placeholder),
   så headern beskriver tabellen korten faktiskt använder.
2. **Sidebar · Lägg till stream**: sökning (`search_tickers`, cache 1h) →
   "Lägg till" hämtar 5 år (`fetch_5y`, cache 1h), sanerar stem (bara a–z0–9 —
   `stream_of` splittar på `_`), kollisionskoll mot fasta + tillagda streams,
   sparar i `session_state["extra_streams"]` → rerun.
3. **Vävning**: finns extra streams anropas `get_data(extra_prices=...)` —
   samma `build_features`, extra-frames behandlas som vilka strömmar som
   helst (fulla suffixet, aldrig till DB). Miss → extra rensas + rerun.
   `register_stream_labels` ger de tillagda samma etikettformat.
4. **Sidebar · Inställningar**: Horisont, Läge, Modell (`key=` så valet
   överlever), hyperparam-multiselect → `params` dict med ENBART aktiva
   knoppar (av = funktionens default).
5. **Urvalskortet**: target (endast tone/`_close`/`_ret`) + stream-pills
   (targetets egen ström bortfiltrerad; per-target `key` bevarar val).
   `feature_cols` = alla kolumner för valda streams.
6. **`prepare_target(df, target, horizon)`** — navet:
   - nivå-target → rörelse (`X_close`→`X_ret`; `tone`→beräknad `tone_diff`,
     vid horisont ≥1 diffas ÖPPNA rader så steget blir Fre→Mån, inte Mån−Sön)
   - därefter horisont-skift av RÖRELSE-kolumnen över handelsdagar
   - ut: `df_h` + frusen `TargetSpec(train_col, level_col, kind, horizon)`.
   Allt modellrelaterat nedanför använder `df_h` + spec; jämför/gruppering
   använder rå-`df` (de beskriver datat, inte prediktionsuppgiften).
7. **Modellen fittas en gång**: `train_model` (`st.cache_resource`,
   max 16) — cache-nyckeln är (df_h, features, train_col, knoppar), så samma
   inställningar = ingen omfit.
8. **Live-kortet** (se §4).
9. **Huvudvyn**: `cached_live` (`st.cache_data`, max 32) →
   `mode.live_panels(df_h, features, spec, …)` → render-fria `Panel`s →
   `render_panel` ritar (line via `line_chart`: Altair utan pan/zoom,
   y-axel utan noll-tvång; tidsspann-trimning med `tail`).
10. **Jämför-kortet**: close-serier indexerade till 100 (rå-`df`).
11. **Evaluering**: `cached_eval` (max 32) → `mode.evaluate` →
    `controller.evaluate/evaluate_direction`: `drop_market_closed` →
    kronologisk split → fit på train → prediktion + naiv baseline på test →
    metrics (+ `add_level_view` för nivå-targets: rekonstruerad nivå vs
    persistence ≡ noll-rörelse). `EvaluationResult.target` bär kolumnnamnet
    så panelerna aldrig gissar det.
12. **Gruppering**: `group_assets` — klustring på 1 − returkorrelation.
13. **Sidebar · chatten**: groundas i `mode.context(result)`.

## 4. Live-fragmentet (var 60:e sekund, bara detta block)

Fragmentets argument är FRUSNA mellan ticks — modellen fittas i steg 7 ovan
och skickas in, så en tick hashar aldrig träningsframen.

```
_live_feature_values(df_rå, feature_cols)
  → get_intraday per ticker (st.cache_data, ttl 60s, per-ticker-nyckel)
  → _ret räknas mot sista lagrade close FÖRE quotens egen börsdag
    (quote-förankrat "idag" — inte användarens lokala kalender)
controller.predict_live(model, df_rå, features, live_values)
  → saknade live-värden faller tillbaka till rå-framens senaste rad
Metric: nivå = _live_level_base ⊕ prognos  (spec.format_move som delta)
  - horisont ≥1: targetens EGEN live-kurs om fast ticker, annars senaste lagrade
  - nowcast: senaste lagrade nivån före quotens börsdag
Graf: daglig förändring — faktisk och prediktion från SAMMA rader/index;
      sista orange punkten = live-prognosen på sin måldag
```

## 5. Cache-karta (vad, nyckel, gräns, ogiltigas av)

| Cache | Typ | Nyckel | Gräns | Ogiltigas av |
|---|---|---|---|---|
| `get_data` | data | extra_prices | ttl 1h | Setup-fyllning (`.clear()`), DB-ändring |
| `get_intraday` | data | ticker | ttl 60s | tid |
| `search_tickers`/`fetch_5y` | data | query/symbol | ttl 1h | tid |
| `train_model` | resource | df_h+features+target+knoppar | 16 | LRU |
| `cached_live`/`cached_eval` | data | mode+df_h+features+spec+params | 32 | LRU |

Modeller finns ALDRIG på disk — bara i `train_model`-cachen, per process.

## 6. Invarianter att inte bryta

- Re-target **före** skift; skift **över handelsdagar**; split **efter** skift.
- Grafer på rörelseskalan — aldrig en rekonstruerad nivåkurva (den ligger på
  facit per konstruktion). Nivån bor i live-siffran + metrics.
- `stream_of` splittar på första `_` → stems är bara alnum; suffixmängden ägs
  av `features.STREAM_SUFFIXES`; `_diff`-suffixet myntas bara av
  `prepare_target` och läses av `evaluation.baseline_kind`.
- Ingen Streamlit i `src/` — modes returnerar `Panel`s, appen ritar.
- Widget-state: explicit `key=` (per-dependency där options byter).
