# 🎬 CineMatch – Empfehlungssystem für Filme

> Ein vollständiges **Machine-Learning-Projekt** in Python: von der Datenanalyse über mehrere Empfehlungsverfahren bis zur Evaluation und einer nutzbaren Empfehlungsfunktion. Das gesamte Notebook ist **auf Deutsch** dokumentiert (Erklärungen, Interpretationen, Fazit).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-green) ![Lizenz](https://img.shields.io/badge/Lizenz-MIT-green)

---

## 📌 Inhaltsverzeichnis

1. [Projektidee](#-projektidee)
2. [Datensatz](#-datensatz)
3. [Methoden](#-methoden)
4. [Ergebnisse](#-ergebnisse)
5. [Projektstruktur](#-projektstruktur)
6. [Installation und Ausführung](#-installation-und-ausführung)
7. [Verwendung der Funktionen](#-verwendung-der-funktionen)
8. [Grenzen und Ausblick](#-grenzen-und-ausblick)

---

## 💡 Projektidee

Streaming-Plattformen und Filmportale wie IMDb müssen aus tausenden Filmen genau die auswählen, die zu einer Person passen. In diesem Projekt wird ein **Recommender System** entwickelt, das auf Basis des Bewertungsverhaltens von 610 Nutzern Filme empfiehlt. Es werden vier Ansätze **implementiert, verglichen und kritisch bewertet**:

| Ansatz | Idee | Personalisiert? |
|---|---|---|
| Beliebtheit (gewichtete Bewertung) | „Was mögen die meisten?“ | ❌ |
| Inhaltsbasiert (Genres, TF-IDF) | „Filme, die inhaltlich ähnlich sind“ | ✅ (pro Film) |
| Item-basiertes kollaboratives Filtern | „Filme, die dasselbe Publikum mag“ | ✅ |
| Matrixfaktorisierung (Biased MF, SGD) | „Latente Geschmacksdimensionen“ | ✅ |

## 📂 Datensatz

Die Daten liegen im Ordner `data/` (Format des bekannten MovieLens-Datensatzes):

| Datei | Spalten | Beschreibung |
|---|---|---|
| `movies.csv` | `movieId`, `title`, `genres` | 9.742 Filme mit Titel (inkl. Jahr) und Genres, getrennt durch `\|` |
| `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` | 100.836 Bewertungen von 610 Nutzern (0,5 – 5,0 Sterne), Zeitraum 1996 – 2018 |

Kennzahlen: Ø Bewertung 3,50 · Matrix zu **98,3 % leer** (Sparsity) · 35 % der Filme haben nur eine Bewertung (Long Tail).

## 🔬 Methoden

Das Notebook `CineMatch_Empfehlungssystem.ipynb` ist wie folgt aufgebaut:

1. **Daten laden und prüfen** – fehlende Werte, Duplikate, Konsistenz der IDs
2. **Aufbereitung** – Jahr aus dem Titel extrahieren, Unix-Zeit in Datum umwandeln, Genre-Listen
3. **Explorative Datenanalyse** – Bewertungsverteilung, Nutzeraktivität, Long-Tail, Sparsity, Genres, Zeitverlauf
4. **Beliebtheitsmodell** – Bayes'scher gewichteter Durchschnitt (wie IMDb-Top-250): `WR = v/(v+m)·R + m/(v+m)·C`
5. **Inhaltsbasiert** – TF-IDF auf Genres + Kosinus-Ähnlichkeit
6. **Item-Based Collaborative Filtering** – nutzerzentrierte Kosinus-Ähnlichkeit (*Adjusted Cosine*) mit Shrinkage, kNN-Vorhersage, Wahl von *k*
7. **Matrixfaktorisierung** – `r̂ = μ + b_u + b_i + p_uᵀq_i`, trainiert mit SGD und L2-Regularisierung, Lernkurve
8. **Evaluation** – 80/20-Split; **RMSE / MAE** (Rating-Vorhersage) sowie **Precision@10 / Recall@10 / Hit-Rate@10** (Top-N)
9. **Finale Funktionen** – Empfehlungen für Nutzer, „Ähnlich wie …“, Cold-Start für neue Nutzer
10. **Fazit, Grenzen, Ausblick**

## 📊 Ergebnisse

**Rating-Vorhersage (Testdaten, niedriger = besser):**

| Modell | RMSE | MAE |
|---|---|---|
| Globaler Mittelwert | 1,024 | 0,818 |
| Film-Mittelwert | 0,937 | 0,726 |
| Bias-Modell (μ + b_u + b_i) | 0,850 | 0,652 |
| Matrixfaktorisierung (f=20) | 0,836 | 0,640 |
| **Item-kNN (k=30)** | **0,827** | **0,624** |

**Top-10-Empfehlungen (Testdaten, höher = besser):**

| Modell | Precision@10 | Recall@10 | Hit-Rate@10 |
|---|---|---|---|
| Zufall | 0,019 | 0,014 | 14,7 % |
| Item-kNN | 0,048 | 0,027 | 26,6 % |
| Matrixfaktorisierung | 0,056 | 0,040 | 31,9 % |
| Hybrid (MF + Beliebtheit) | 0,091 | 0,070 | 46,0 % |
| **Beliebtheit (gewichtet)** | **0,108** | **0,085** | **50,9 %** |

**Kernaussagen**

* Kollaboratives Filtern senkt den Vorhersagefehler gegenüber allen einfachen Baselines.
* Beim Top-N-Ranking ist die **Beliebtheit** in diesem Offline-Test schwer zu schlagen – ein typischer Effekt bei kleinen MovieLens-Daten. Die personalisierten Modelle liegen aber klar über dem Zufall.
* Ein **Kandidatenfilter** (nur Filme mit ≥ 30 Bewertungen) ist wichtig, damit nicht Nischenfilme mit wenigen Zufallstreffern empfohlen werden.

> Alle Zahlen stammen aus dem ausgeführten Notebook (feste Zufallszahl, `SEED = 42`) und sind reproduzierbar.

## 🗂 Projektstruktur

```
CineMatch/
├── CineMatch_Empfehlungssystem.ipynb   # Hauptnotebook (deutsch, mit Ausgaben)
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## ⚙️ Installation und Ausführung

```bash
git clone https://github.com/0Zyyra0/CineMatch.git
cd CineMatch

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook CineMatch_Empfehlungssystem.ipynb
```

Das Notebook muss **im Projektordner** gestartet werden, damit die relativen Pfade `data/…` stimmen. Die Laufzeit beträgt ca. 1 Minute (das SGD-Training der Matrixfaktorisierung dominiert).

## 🧩 Verwendung der Funktionen

Nach dem Ausführen des Notebooks stehen folgende Funktionen bereit:

```python
empfehle_fuer_nutzer(user_id=1, n=10)          # personalisierte Top-N (Matrixfaktorisierung)
empfehle_aehnlich("Matrix, The (1999)", n=8)   # kollaborativ: "Ähnlich wie ..."
inhaltsbasiert("Toy Story (1995)", n=8)        # Genre-Ähnlichkeit (auch für unbewertete Filme)
empfehle_neuer_nutzer(["Sci-Fi", "Thriller"])  # Cold-Start: Beliebtes im Wunschgenre
finde_titel("matrix")                          # Titelsuche, um exakte Titel zu finden
```

## 🚧 Grenzen und Ausblick

**Grenzen**

* Kleiner Datensatz, nur explizite Bewertungen
* Ein einzelner zufälliger Split; Hyperparameter ohne separates Validierungsset gewählt
* Inhaltsmodell nutzt ausschließlich Genres
* Offline-Metriken zählen unbewertete, aber gute Empfehlungen als „falsch“

**Ausblick**

* Kreuzvalidierung, Grid-Search, zeitbasierter Split
* Hybridmodelle und zusätzliche Metadaten (Schauspieler, Regie, Handlung)
* ALS / Neural Collaborative Filtering, implizites Feedback
* Weitere Metriken: NDCG, Coverage, Diversität, Novelty
* Deployment als Web-App (Streamlit)

## 📄 Lizenz

MIT-Lizenz, siehe [LICENSE](LICENSE). Die Daten stammen aus einem öffentlich verfügbaren Filmbewertungsdatensatz im MovieLens-Format und dienen hier Lern- und Demonstrationszwecken.
