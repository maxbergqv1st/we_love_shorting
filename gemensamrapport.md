# Huvudresultat.
  Vår hypotes är att dagens marknadspriser (där vi valt S&P 500, OMX30, EURO STOXX 50, guld, silver, koppar och råolja) påverkas av nyhetsflödets sentiment som vi kallar “tone” och att detta samband kan användas som en shorting-signal. 
  Modellen tränas för att prediktera dagens eller nästa dags priser eller ton (avkastning/dagsförändring), inte nivån i sig då det blev fel siffror att läsa av jämfört med avkastning. 

  Utvärderingsmetod: Datan delas kronologiskt i tränings- och testperiod. Modellens prestanda jämförs alltid mot en naiv baseline och gårdagens rörelse för return-mål, eller medelvärdet för tone-målet. Samt mot majoritetsklassen för riktningsklassificering (upp/ner/oförändrad).

  Tolkning: Baseline slår våran modell då den är naiv och att prediktera finans är otroligt svårt och har stora hopp som är svåra att läsa av.


  Modellvalet (linjär regression/Ridge vs. Random Forest) testades via hyperparametrarna. Feature-vikterna visade sig att ju fler fonder/ädelmetaller mm vi la till desto bättre blev modellen så länge den var någorlunda passande till områdena vi redan lagt in.

  Sammantaget stöder resultaten hypotesen att marknadspriser innehåller prediktiv information om nyhetssentiment, men dock inte tillräckligt bra för att med säkerhet kunna prediktera resultat och ska inte användas för att investera.

# Bakgrund 
Projektuppgiften krävde ett komplett AI/ML-projekt med tre delar: datalagring i en databas, AI modellering i Python, och ett Streamlit gränssnitt. Istället för ett färdigt Kaggle dataset valde gruppen att arbeta med verklig, löpande data hämtad via öppna API:er. Dels för ett mer autentiskt fullstack flöde med datainsamling, felhantering och cachning men också för att arbeta med en fråga där utfallet inte redan var givet.

Efter en ganska bred idégenomgång och brainstorm, där vi bland annat övervägde väderdata, sportresultat och samhällskritiska teman som brottsstatistik, landade gruppen i en finansiell frågeställning: finns det korrelation mellan hur negativt eller positivt nyhetsrapporteringen är och kursutvecklingen på finansiella instrument? Hypotesen var att daglig nyhetston, mätt via GDELT:s (Global Database of Events, Language, and Tone) sentimentanalys, kunde sättas i relation till prisrörelser hos ett antal finansiella instrument. Vi var medvetna om att ett starkt samband sannolikt redan är känt och utnyttjat av marknaden själv, men syftet var att undersöka frågan metodiskt snarare än att förvänta oss ett tydligt resultat.

Tidigt i arbetet uppstod frågan om projektet riskerade att röra sig mot tidsserieanalys, vilket ligger utanför kursens innehåll, eftersom både nyhetston och marknadspriser är tidsordnad data. Efter avstämning konstaterade vi att så länge målvariabeln (tone) predikteras utifrån externa variabler, och aldrig utifrån sina egna tidigare värden, rör vi oss fortfarande inom klassisk regression, men med extra metodologisk noggrannhet: kronologisk uppdelning av tränings- och testdata, samt användning av procentuell förändring snarare än råa prisnivåer, för att undvika skenbara samband som egentligen bara speglar en delad trend över tid.

Initialt var målet att arkitekturen skulle hämta daglig nyhetston via GDELT:s öppna API och prisdata för sju finansiella instrument (S&P 500, OMX Stockholm 30, EURO STOXX 50, guld, silver, koppar och olja) via Yahoo Finance, populera en SQLite databas, och erbjuda ett Streamlit gränssnitt där användaren fritt kan välja målvariabel och prediktorer, träna en linjär regressionsmodell samt presentera resultatet mot en naiv baseline.

# Kort teknisk specifikation kring diverse teknologi som har använts.

Flödet mellan filerna 
Data vandrar genom filerna i tur och ordning, och varje fil har sitt eget jobb: källor → databas → features → modell → controller → app.
Först hämtas data. sources/yahoo.py hämtar priser (S&P 500, OMX30, EURO STOXX 50, guld, silver, koppar, olja) från Yahoo Finance, och sources/gdelt.py hämtar daglig nyhetston från GDELT. Sedan sparas allt i db.py, som lägger datan i en SQLite-databas.

Därefter bearbetar features.py rådatan till det modellen faktiskt tränar på: procentuell förändring, glidande medelvärden och volatilitet.

Sedan tränas och mäts modellen. signal_model.py tränar AI-modellerna, och evaluation.py jämför modellen mot en naiv baseline för att se hur bra den egentligen är.

controller.py är limmet som binder ihop allt och anropar källor, databas, features och modell i rätt ordning. analysis.py bestämmer vilken sorts analys som körs (regression, riktning eller gruppering).

Till sist visas resultatet. app.py är gränssnittet byggt i Streamlit, där användaren väljer målvariabel och prediktorer och ser resultatet. chatbot.py är en AI-chatt som svarar på frågor.

Alla tekniker och vad de används till

- Python 3.14 är språket hela projektet är skrivet i.

- yfinance används för att hämta prisdata från Yahoo Finance, och urllib (inbyggt i Python) för att hämta nyhetston från GDELT.

- SQLite via sqlite3 (inbyggt) är databasen som lagrar all data, alltså backend-delen.

- pandas hanterar tabelldata, feature engineering och train/test-uppdelningen, medan numpy sköter de numeriska beräkningarna bakom.

- scikit-learn är kärnan i AI/ML-delen och används för linjär regression, Ridge, Random Forest, logistisk regression och klustring.

- Streamlit är frontend-ramverket som bygger gränssnittet, och Altair ritar diagrammen i appen.

- OpenRouter driver AI-chatten som svarar på frågor.

För kodkvalitet under utvecklingen använde vi ruff (formatering och lint), pytest (tester) och mypy (typkontroll). Git och GitHub sköter versionshantering och samarbete, och watchdog låter Streamlit ladda om appen automatiskt när koden ändras.

# Utvärdering av gruppens arbete (vad har varit bra, vad har ni lärt er, hur har arbetet med Git)
Vi använde GitHub med branches, vilket funkade jättebra — inga konflikter eller regressioner uppstod, och det verkar som att allt vi har lärt oss föll på plats och gav ett bra resultat. Varje ny funktion (till exempel UI-fräschningen, train/test-utvärderingen och chatboten) byggdes i sin egen branch innan den slogs ihop med main, så vi kunde jobba parallellt utan att köra över varandra. Vi kodgranskade varandras PRs utan krockar, och vi var tydliga och försiktiga i den biten som vi alla gjort separat — några PRs innehöll till och med konkreta granskningskommentarer som fixades innan merge, vilket kändes som riktigt utvecklingsarbete snarare än bara en skoluppgift. Vi gitignorade rätt filer (bland annat databasen, hemligheter och cache-mappar) och hade flytande kommunikation i vår stängda Discord-grupp. Namnkonventionen (till exempel "feature/...", "fix/...") accepterades tyst av alla utan problem, vilket gjorde att branchlistan aldrig blev rörig. Vissa överraskande moment dök upp då och då, till exempel en konkret utmaning där en körande Streamlit-process kunde bli inaktuell på grund av caching av gammal kod — i så fall behövde man köra Streamlit igen manuellt i terminalen. Det var en bra lärdom kring lokal utvecklingsmiljö och en allmänt nyttig erfarenhet inför framtiden, och något vi antagligen stöter på igen i kommande projekt.

Dependency-hanteringen och hela setupen var tydligt dokumenterad i READMEn, så det var enkelt att komma igång även när man hoppat av projektet ett par dagar och behövde uppdatera sin lokala miljö. Vikten av train/test-split uppgraderades under projektets gång med evaluation.py, och i allmänhet arbetade vi iterativt hela tiden, precis som vi har lärt oss i skolan — vi byggde en enkel version först och förbättrade sedan steg för steg istället för att försöka göra allt perfekt från början.

Under flera veckors gång har gruppens arbete och samarbete mellan oss studenter fungerat alldeles perfekt: dagliga standups, kommunikation, skojiga moment, förståelse för varandras situation, flexibilitet och kompatibilitet mellan medlemmarna på högsta nivån med mera. Alla dessa förutsättningar bidrog till att projektet lyckades i slutändan, och vi kände oss aldrig oroliga för att fråga varandra om hjälp när något krånglade. Standupsen gjorde att vi tidigt upptäckte om någon höll på att bygga samma sak, eller om en idé redan täcktes av något annat i koden, vilket sparade oss från dubbelarbete flera gånger under projektet.

Om vi hade gjort om projektet hade vi nog lagt lite mer tid på planering i början, innan vi satte igång med kodandet, för att säkerställa att alla hade samma bild av vad som skulle byggas och i vilken ordning. Inget av detta blev ett riktigt problem för oss, men det hade gjort arbetsflödet ännu smidigare från dag ett.

