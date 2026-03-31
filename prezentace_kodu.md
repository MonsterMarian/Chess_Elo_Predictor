# Tahák pro prezentaci projektu Elo Predictor

Tento dokument je tvůj návod, jak jednoduše a srozumitelně odprezentovat projekt. Vysvětluje jak samotné fungování pod pokličkou, tak i byznysový a praktický přesah aplikace.

---

## 1. Úvod (O čem projekt je a proč vůbec vznikl)
**Co říct:** 
> "Vytvořil jsem aplikaci, která funguje jako váš osobní šachový věštec. Dokáže odhadnout, jaký šachový rating (Elo) budete mít na Chess.com za měsíc, nebo za jak dlouho pravděpodobně dosáhnete svého vysněného cíle (např. 2000 Elo). Aby to fungovalo, program se učil na historii více než 1500 skutečných šachistů z celého světa."

---

## 2. Proč by za to někdo měl platit? (Byznys a přidaná hodnota)
*Když se tě zeptají (nebo jako součást hlavní řeči), jaký to má smysl:*

**Co říct:**
> "Chess.com sice hráčům ukazuje jejich aktuální rating, ale neřekne jim, co je čeká. Strašně moc šachistů kupuje drahé kurzy a tréninky, protože se chtějí zlepšit. 
> 
> Tento nástroj jim za drobný poplatek (nebo jako prémiové předplatné) dává matematicky podložené zrcadlo: Ukáže jim, jestli jejich aktuální trend vůbec vede k cíli, jak velké mají výkyvy výkonnosti (nejistota) a přesně za jak dlouho – při stejném tempu – dosáhnou další výkonnostní třídy. Lidé platí za jistotu a za to, že vidí svůj progres zasazený do reálného časového rámce."

---

## 3. Co byla nejtěžší věc na vývoji?
*Prokázání, že to nebyla jen "hříčka na odpoledne".*

**Co říct:**
> "Zdaleka nejtěžší na celém projektu nebyl samotný web, ale **zpracování dat a matematické rozdělení hráčů**.
>
> 1. Museli jsme stáhnout statisíce záznamů o tom, kdy a jak který hráč hrál. Data bývají neúplná, hráči mají výpadky (hrají měsíc v kuse a pak půl roku ne) – naučit počítač toto pochopit bylo extrémně složité. **Jak jsme to vyřešili?** Zavedli jsme takzvané *Měsíční snapshoty*. Místo milionů jednotlivých her a datového zmatku ukládáme do databáze jen shrnutí za každý měsíc (konečné Elo v daném měsíci, počet her, a délky pauz). Tím jsme obrovský chaos zredukovali na čistou měsíční osu, ze které umělá inteligence snadno čte trendy.
> 2. Druhá obrovská výzva byla, že **nelze porovnávat začátečníka s pokročilým**. Začátečník s 800 Elo může vyrůst o 200 bodů za měsíc, zatímco expert blížící se hranici 2000 Elo roste třeba jen o 10 bodů za půl roku. Proto jsme museli celou umělou inteligenci rozdělit na několik samostatných "mozků" (Modelů) podle výkonnostních úrovní pro hráče od 600 do 2000 Elo (tzv. "Bands"), aby dávala realistické výsledky pro každého."

---

## 4. Jak to uvnitř vlastně funguje (Přirovnání k restauraci)
Při ukazování technické stránky projektu nepoužívej moc složitých pojmů. Můžeš to přirovnat k fungování restaurace:

### A) `downloader.py` (Zásobování surovin)
*   "Připojí se k veřejným datům Chess.com a automaticky stáhne obrovskou historii šachistů. Očistí je a uloží do místního skladu (lokální databáze), abychom data nemuseli pokaždé lovit znovu na internetu."

### B) `predictor.py` (Šéfkuchař / Mozek inteligence)
*   "Zde v pozadí sídlí model strojového učení (algoritmus jménem LightGBM). Nejdřív si vezme suroviny (data z databáze) a **'učí se'** z nich, jak rostou různé typy hráčů. Zkoumá stabilitu her, výkyvy formy a kadenci hraní. A když mu poté zadáme nového hráče, matematicky nasimuluje jeho vývoj v dalších měsících."
*   **Proč LightGBM a ne pouze obyčejné K-NN?** (Kdyby se někdo ptal na princip)
    *   *Běžné K-NN (K-Nejbližších sousedů)* by fungovalo tak, že vezme vaše aktuální Elo a najde v databázi 5 lidí, kteří měli v minulosti podobné skóre. Následně udělá průměr toho, jak dopadli oni. Je to sice jednoduché, ale extrémně naivní – nezajímá ho totiž, že vy hrajete 10x častěji než oni, a zprůměruje míchání "jablek a hrušek".
    *   *LightGBM (Stromový gradientní boosting)* nehledá pouhé "sousedy", ale **učí se chování**. Vytváří složité stromy rozhodnutí (např. "Pokud má hráč vysokou kadenci hraní, ale jeho skóre neustále lítá nahoru a dolů, jeho budoucí růst bude velmi malý"). Dokáže pochopit i to, že vliv nějakého faktu (např. herní výpadek) dopadá na hráče s 800 Elo úplně jinak než na hráče s 1800 Elo. Díky tomu je obrovsky přesnější a matematicky dospělejší.

### C) `app.py` (Číšník / Server)
*   "Kód, který to celé drží pohromadě (webový server). Když kliknete na tlačítko, on ten úkol vezme, zeptá se Chess.com na vaši aktuální formu, letí to ukázat 'Šéfkuchaři' (`predictor.py`), sebere od něj křivku budoucího vývoje a donese ji uživateli."

### D) `templates/index.html` (Stůl v restauraci / Vizualizace)
*   "Samotná viditelná stránka na monitoru. Moderní a plynulé zpracování. Převezme surová čísla (predikci) a vykreslí z nich pěkné vizuální kartičky a odhadované mílníky (Milestones)."

---

## 5. Průběh Ukázky naživo (Live Demo)
Až to budeš prezentovat, proveď přesně tohle:

1. **Otevři prohlížeč** na adrese `http://127.0.0.1:5000` (musíš mít aplikaci předem zapnutou přes terminál `python app.py`).
2. **Představ design:** "Tady je jednoduché vyhledávání. Žádné složité registrace, napíšete jméno a hotovo."
3. **Zadej účet:** Napiš tam třeba `"gothamchess"` ukaž výběr módů (Blitz/Rapid) a klikni na **Analyze Player**.
4. **Okometnuj vteřinu načítání:** "Server právě běží na Chess.com, stahuje poslední výsledky hráče, a zkoumá jeho formuli přes naši umělou inteligenci."
5. **Vysvětli výsledky:** 
   * Ukaž barevné odznaky (jistota odhadu - "Vidíme, jak moc si je algoritmus predikcí jistý").
   * Ukaž políčka **Predikce** a grafického **Percentilu**.
   * Úplně nakonec ukaž **Milestones (Mílníky):** *"Tady mu přímo říkáme – do dvou měsíců mít 1800 Elo fakt nedáš, protože tvoje křivka roste příliš pomalu. Ale např. 1750 zvládneš za půl roku."*
