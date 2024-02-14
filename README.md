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
2) Lokalisierung der 4 Bohrlöcher
   1) Anahnd der Abständer Skalierung und Rotation berechnen und so den zu extrahierenden Bereich ermitteln
   2) Frage: Wie zuverlässig?
   3) Alternativ: Andere Komponenten als Orientierungspunkte erkennen
3) Ein SVM-Modell trainieren das PCB von Hintergrund unterscheiden kann
   1) Konturen finden und dann kleinstmögliches Rechteck bestimmen
   2) Mögliche Herausforderung: Pins sind nicht sehr gut von Hintergrund zu unterscheiden
      1) Eventuell kann auch hier der Trick angewedente werden nur das PCB zu erkennen und dann "manuell" um die nötige Größe nach oben zu verlängern um auch die Pins mit im Bild zu haben