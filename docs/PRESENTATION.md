# Presentation — we_love_shorting

Manus och tankemodell för milstolpspresentationen (v.39). Uppgiftens tre delar:
**databas** (SQLite) + **ML** (scikit-learn) + **Streamlit**-frontend.

## Hisspitchen (30 sekunder)

> Vi testar hypotesen att marknadspriser och nyhetssentiment bär information om
> varandra. Vi hämtar sju prisströmmar och GDELT:s nyheston, lagrar i databas,
> tränar en modell interaktivt, och mäter den ärligt mot naiva baselines — allt
> i en dashboard där **vilken ström som helst kan vara målet**.

## Arkitekturen — så ska man tänka

Envägsflöde i lager; varje lager känner bara lagret under sig. Flera personer
kan jobba i varsitt lager utan att krocka, och allt under `app.py` är
enhetstestat utan UI.

```
sources/gdelt.py ─┐  hämtning (urllib, retry/backoff på 429)
sources/yahoo.py ─┤
                  ▼
db.py             SQLite-arkiv: whitelistade tabeller, idempotent upsert,
                  inkrementell påfyllning — DB:n är enda sanningskällan
                  ▼
features.py       MÖTESPUNKTEN: varje ström beräknar ret/ma/vol på sin EGEN
                  handelskalender, sedan outer-merge på datum + ffill;
                  market_closed flaggar carry-forward-rader
                  ▼
signal_model.py   sklearn: OLS / Ridge / Random Forest + riktningsklassificerare
evaluation.py     kronologisk split, baselines, metrics
                  ▼
controller.py     dirigenten: prepare_target, run, evaluate, predict_live
                  ▼
analysis.py       STRATEGIMÖNSTRET: modes returnerar render-fria Panels —
                  testbara utan Streamlit; ny mode = en registry-rad
                  ▼
app.py            enda Streamlit-filen: ritar Panels, äger ingen domänlogik
```

## De tre idéerna värda att förklara på djupet

### a) Träna på förändringen, inte nivån
En stängningskurs är nästan en random walk — tränar man på nivån lär sig
modellen bara "imorgon ≈ idag". Ett nivå-target re-targetas därför internt till
sin dagliga rörelse (`controller.prepare_target`); graferna visar
förändringsskalan (prediktionens tecken = upp/ner-beskedet) och nivån visas
bara där den är ärlig: live-siffran och metrics-tabellen. En rekonstruerad
nivåkurva ligger på den faktiska *per konstruktion* (persistence, inte
skicklighet) — därför ritas den aldrig.

### b) Ärlig evaluering
Kronologisk split (aldrig shuffle — autokorrelerade serier läcker), alltid mot
en naiv baseline: historiskt snitt för rörelser, persistence för nivåer.
Elegant symmetri: *persistence på nivåskalan är exakt samma sak som att gissa
noll rörelse* — modellen slår persistence på kursen precis när den slår "gissa
alltid 0" på förändringen. **Modellen slår inte alltid baseline — att
dashboarden visar det ärligt är en feature, inte ett misslyckande.**

### c) Interaktivt, aldrig persisterat
Modeller tränas om vid varje ändrad inställning (cachas i minnet med tak,
aldrig till disk) — ingen inaktuell modell på disk, auto-eval utan "kör"-knapp.

## Demo-manus (i den här ordningen)

1. **Setup-dialogen** → påfyllning + "Ladda data" — *"databasen är enda
   källan; nätverket är frikopplat från träningen"*.
2. **Defaultvyn** (tone som target): topbar med dataspann, target- och
   stream-pills — *"vilken ström som helst kan vara målet; targetets egen
   ström exkluderas automatiskt"*.
3. **Byt target till Guld, Horisont → Imorgon** — *"skiftet sker över
   handelsdagar, annars lär sig modellen att fredag = lördag"*.
4. **Live-kortet** — nivåprognos med delta, minutuppdatering; grafen visar
   daglig förändring, faktisk vs prediktion på samma dagar — glappet är
   modellens miss.
5. **Hyperparametrar**: toggla Random Forest + max_depth, hovra
   (?)-tooltipen — *"en registry-rad per ratt; av = modellens default"*.
6. **Evaluering**: dubbla skalorna i tabellen, held-out-grafen, viktvyn
   (*"noll extra fits — vikterna följer med evalueringen"*),
   residual-histogrammet.
7. **Riktning med flat_frac = 0** — *"tvunget upp/ner-val varje dag, mätt
   mot majoritets-baseline"*.
8. **Lägg till en stream live** (välj något med lång historik, t.ex. AAPL —
   inte en ny notering: tabellen trunkeras till kortaste historiken!) — den
   vävs in med alla kolumner och tränas direkt, utan att röra databasen.
9. **Gruppering**: heatmapen — *"ädelmetallerna klustrar ihop; korrelation på
   returer, aldrig på nivåer"*.
10. **AI-chatten** som avslutning — groundad i den aktuella evalueringen.

## Frågor vi lär få — och ärliga svar

- **"Funkar det? Tjänar man pengar?"** — "Oftast slår modellen inte
  persistence — väntat för nästan-random-walks. Poängen med bygget är att man
  ser det ärligt i stället för att luras av en snygg kurva."
- **"Varför ser prediktionslinjen flack ut?"** — "Modellen säger nära noll när
  signalen är svag — det är information, inte feghet."
- **"Hur vet ni att koden håller?"** — 44 tester inkl. headless AppTest-körning
  av hela appen (noll varningar tillåtna), ruff + mypy i grönt, och två
  multiagent-granskningsomgångar vars fynd är åtgärdade.
- **"Kända begränsningar?"** — market_closed är en proxy mot EN
  referenskalender; extra streams får inga live-kurser och trunkerar tabellen
  till sin egen historik; live-hämtningen är sekventiell per ticker (batchbar).

## Köra själv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app.py
```
