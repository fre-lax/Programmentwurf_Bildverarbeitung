def remove_comments(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = [line.split('#', 1)[0] + '\n' for line in lines]

    with open(filename, 'w') as f:
        f.writelines(lines)


def entferne_leerzeilen(input_datei, output_datei):
    with open(input_datei, 'r', encoding='utf-8') as datei:
        zeilen = datei.readlines()

    with open(output_datei, 'w', encoding='utf-8') as neue_datei:
        for zeile in zeilen:
            if zeile.strip():  # Prüft, ob die Zeile nicht leer ist
                neue_datei.write(zeile)

remove_comments('./Programmentwurf_Abgabe_ohne_kommentare.py')
entferne_leerzeilen('./Programmentwurf_Abgabe_ohne_kommentare.py', './Programmentwurf_Abgabe_ohne_kommentare.py')