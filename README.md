# Programmentwurf_Bildverarbeitung

Gruppe: Luca Müller, Jonas Hinkel, Jonas Ledig, Frederic Stepp

## Aufgabe: Erkennen und Ausschneiden von PCB aus Bildern und in Ordner einordnen
Aufgabenstellung auf [Website zur Vorlesung](https://virtlab.fakultaet-technik.de/Vorlesungen/Bildverarbeitung.html#/Programmentwurf) zu finden

Abgabe: 20.02.2024, online


##### Problem: PCB sind wesentlich inhomogener als Cashew-Nüsse

Bilder hier zu finden: [Google Drive]()

Lösungsideen

1) Kantenerkennung (PCB ist Recheckig)
   1) Mit Kanten Konturen finden, um Konturen kleinstmögliches Rechteck legen
   2) Pins des PCB sind sehr dünn, haben ähnliche Farbe wie Hintergrund
   3) Manche PCB haben deutlich helle Kanten auf allen vier Seiten (gut) andere sind nicht so gut zu Erkennen
   4) Vor Kantenerkennung ggf nach Farben Filtern
2) Lokalisierung der 4 Bohrlöcher
   1) Anahnd der Abständer Skalierung und Rotation berechnen und so den zu extrahierenden Bereich ermitteln
   2) Frage: Wie zuverlässig?
   3) Alternativ: Andere Komponenten als Orientierungspunkte erkennen
3) Ein SVM-Modell trainieren das PCB von Hintergrund unterscheiden kann
   1) Konturen finden und dann kleinstmögliches Rechteck bestimmen
   2) Mögliche Herausforderung: Pins sind nicht sehr gut von Hintergrund zu unterscheiden
      1) Eventuell kann auch hier der Trick angewedente werden nur das PCB zu erkennen und dann "manuell" um die nötige Größe nach oben zu verlängern um auch die Pins mit im Bild zu haben


###### 1) Kantenerkennung

Erste Kantenerkennung:
![alt text](image.png)

Update mit Gauss-Weichzeichner und anderen Einstellungen:
![alt text](image-1.png)

Erstelle Konturen von Kantenerkennung und Lege ein Rechteck darum
![alt text](image-2.png)
![alt text](image-3.png)

Wie zu sehen: Noch nicht zuverlässig. Bei den meißten Bildern klappt es gut, aber noch nicht immer.
Außerdem sind es noch mehrere Rechtecke

Zu Beginn wurde eine Blaumaske hinzugefügt:
Bildbereiche mit den Bedingungen (b >120) & (r>10) werden auf weiß gesetzt, der recht auf schwarz
![alt text](image-4.png)

Danach obige Schritte führt zu guten Ergebnissen:
![alt text](image-5.png)

Das Rechteck wird ins urpsrüngliche Foto eingezeichnet:
![alt text](image-6.png)
Und zugeschnitten/ gedreht:
![alt text](image-7.png)

Bei der Batch-Bearbeitung fielen weitere Fehler auf. Fast ausschließlich bei anomalen PCBs.
Vermutlich insbesondere durch die Blau-Maske
![alt text](image-8.png)

Veränderung der Parameter für Blaumaske und gausscher Weichzeichner vor der Blaumaske fürten zu starker Verbesserung der Ergebnisse
