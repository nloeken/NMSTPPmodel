import os
# Pandas wird hier nicht mehr direkt benötigt, da wir das DataFrame nicht mehr handhaben
from data_loader import load_and_merge
from data_preprocessing import preprocess
from config import COMBINED_FILE

def main():
    
    print(">>> Schritt 1: Starte das Laden und Zusammenführen der Rohdaten...")
    # Diese Funktion liest die JSON-Dateien, mergt sie und speichert für jedes Spiel eine CSV im MERGED_DIR
    load_and_merge()
    print(">>> Schritt 1: Laden und Zusammenführen der Rohdaten abgeschlossen.")
    print("-" * 50)

    print(f">>> Schritt 2: Starte die Erstellung der finalen Datei '{COMBINED_FILE}'...")
    # Diese Funktion liest jetzt alle einzelnen CSVs, verarbeitet sie speichereffizient
    # und speichert das Ergebnis direkt in der Zieldatei COMBINED_FILE.
    # Sie gibt nichts (None) zurück, da die Speicheroperation intern stattfindet.
    preprocess()
    print(f">>> Schritt 2: Finale Datei erfolgreich unter '{COMBINED_FILE}' erstellt.")
    print("-" * 50)
    
    # Schritt 3 ist nicht mehr nötig, da das Speichern jetzt Teil von Schritt 2 ist.
    
    print("Alle Prozesse erfolgreich abgeschlossen!")


if __name__ == "__main__":
    main()