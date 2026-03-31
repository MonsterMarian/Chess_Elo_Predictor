# ♟️ Chess Milestone Predictor (Osobní AI šachový věštec)

**Chess Milestone Predictor** je webová aplikace s umělou inteligencí, která analyzuje vaši šachovou historii a s vysokou přesností predikuje, jaké bude vaše hodnocení (Elo) v budoucnosti. 

Zatímco tradiční statistiky vám řeknou jen to, jak hrajete dnes, náš nástroj funguje jako **osobní analytik**. Zodpoví vám otázky typu: *Zlepšuji se dostatečně rychle? Zabírá můj trénink? Za jak dlouho s aktuálním tempem dosáhnu 2000 Elo?*

---

## 💎 Proč je tento nástroj užitečný? (Přidaná hodnota)

Mnoho šachistů investuje nemalé peníze do kurzů a tréninků, protože se chtějí zlepšit. Náš systém dává těmto hráčům **matematicky podložené zrcadlo**:
- **Simuluje budoucnost:** Na základě aktivity, výkyvů formy a historie spočítá, jaké Elo budete mít za měsíc či půl roku.
- **Odhaluje milníky (Milestones):** Přesně určí, kolik měsíců vám bude trvat dosažení dalších výkonnostních skoků (např. +100 Elo).
- **Varuje před stagnací:** Pokud váš aktuální herní trend nikam nevede (stagnujete nebo klesáte), algoritmus vás na to upozorní.

*(Díky tomu projekt otevírá obrovský byznysový potenciál u ambiciózních hráčů hledajících motivaci a hmatatelné cíle na své cestě.)*

---

## 🧠 Jak to funguje pod pokličkou?

Celý projekt se skládá ze čtyř samostatných částí. Pro zjednodušení si to lze představit jako fungování restaurace:

1. **`downloader.py` (Dodavatel dat):** Automaticky stahuje statisíce odehraných her z Chess.com a ukládá z nich shrnutí (snapshoty) do lokální databáze (`elo_history_v3.db`).
2. **`predictor.py` (Šéfkuchař / Mozek inteligence):** Umělá inteligence (LightGBM). Tato část se učí ze stažených dat. Protože začátečník se zlepšuje úplně jinak než velmistr, je model chytře rozdělen do několika úrovní (tzv. *Elo Bands*: 600–1000, 1000–1400, atd.).
3. **`app.py` (Číšník / Server):** Webový server (Flask). Neustále čeká na dotazy uživatelů. Jakmile zadáte své jméno, stáhne bleskově vaši aktuální formu, zeptá se *Mozku* na predikci a výsledek donese na váš stůl.
4. **`templates/index.html` (Jídelní stůl / Vizualizace):** Samotná webová stránka s moderním vzhledem. Převezme surová data a vykreslí z nich pěkné vizuální kartičky a odhadované mílníky vyhledaného hráče.

---

## 🚀 Rychlý start (Jak aplikaci spustit)

Pokud si chcete projekt spustit lokálně, postupujte krok za krokem:

### 1. Instalace závislostí
Otevřete terminál ve složce projektu a nainstalujte potřebné knihovny:
```bash
pip install -r requirements.txt
```

### 2. Sběr dat do databáze
Databáze se musí nejprve naplnit živými historickými údaji. (Stahuje hráče s Elo **600–2000**).
```bash
python downloader.py --limit 5000
```

### 3. Trénink Umělé inteligence (Mozku)
Jakmile narazíte alespoň na pár stovek hráčů, musíte natrénovat modely pro různé herní formáty (doporučeno spustit po stažení dostatku dat):
```bash
python predictor.py --type blitz
python predictor.py --type rapid
python predictor.py --type bullet
```
*(Natrénované modely se uloží do složky `models/`)*

### 4. Spuštění samotného webu
Zapněte vizuální aplikaci pro uživatele:
```bash
python app.py
```
Nyní stačí otevřít internetový prohlížeč a přejít na adresu: **http://127.0.0.1:5000**

---

## 🏆 Největší technické výzvy a jejich řešení
Pokud by vás zajímalo, co byla na vývoji absolutně nejtěžší část: Nebyl to web, ale **čištění obrovského množství reálných dat a pochopení času**. 
Archivy zápasů na Chess.com jsou gigantické a hráči hrají velmi nepravidelně (někdy 100 her za víkend, někdy měsíc vůbec nic). Naučit AI zpracovávat "výpadky formy" jako relevantní matematický znak a striktně rozčlenit učení podle dosažené Elo úrovně zabralo nejvíce analytického úsilí.

**Jak jsme tento problém vyřešili?**
Místo abychom stahovali každou jednolivou hru a AI se v těch milionech mikrozměn utopila, zavedli jsme **Měsíční snapshoty (shrnutí)**. Algoritmus analyzuje historii měsíc po měsíci. Pro každý měsíc si uloží jen konečné Elo a aktivitu (počet uhraných her). Program z toho následně tvoří chytré statistiky jako např. `max_gap` (nejdelší souvislá pauza v hraní) nebo `volatility_3m` (jak moc Elo hráči skákalo za poslední 3 měsíce). Tím se nesmyslně obrovská a chaotická data transformovala do čisté, stabilní časové osy, které náš model perfektně rozumí.

---
> ⚠️ **Limity projektu:** Aplikace s jistotou odhaduje pouze hráče s Elo **od 600 do 2000**. Pro začátečníky s Elem pod 600 nebo naopak velmistry nad 2000 systém záměrně zablokuje predikci, protože pro ně nemá dostatek natrénovaných dat k doručení perfektně přesné odpovědi. Úloha rovněž předpokládá výhled maximálně 6-12 měsíců dopředu.
