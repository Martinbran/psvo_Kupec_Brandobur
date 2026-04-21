# Postup – Zadanie 4: 3D Gaussian Splatting

## 1. Príprava dát
Na vlastnú scénu som použil model autíčka **Auticko_3**.

Fotografie som **zmenšil pred spracovaním**, aby COLMAP aj tréning bežali rozumne rýchlo. Výsledné rozmery obrázkov boli:

- **šírka:** 960 px
- **výška:** 1280 px

Toto zodpovedá aj vlastnostiam výsledných JPEG obrázkov.

---

## 2. Spustenie Docker prostredia
Najprv som spustil pripravené Docker prostredie cez:

```bash
./run.sh
```

Po spustení som sa dostal do kontajnera do priečinka:

```bash
/opt/gaussian-splatting
```

---

## 3. Príprava datasetu
Fotografie som uložil do štruktúry:

```bash
data/auticko_3/input
```

Počet pripravených snímok bol približne **159**.

Zároveň som vytvoril priečinok pre COLMAP výstup:

```bash
data/auticko_3/distorted/sparse/0
```

---

## 4. COLMAP – vytvorenie projektu
V kontajneri som spustil:

```bash
colmap gui
```

V COLMAP som vytvoril nový projekt cez:

```text
File → New project
```

Použité cesty:

- **Database path:**
  ```bash
  /opt/gaussian-splatting/data/auticko_3/database.db
  ```
- **Image path:**
  ```bash
  /opt/gaussian-splatting/data/auticko_3/input
  ```
- **Project path:**
  ```bash
  /opt/gaussian-splatting/data/auticko_3/distorted/sparse/0/project.ini
  ```

---

## 5. COLMAP – Feature extraction
V COLMAP som spustil:

```text
Processing → Feature extraction
```

Použité nastavenia:

- **Camera model:** `SIMPLE_PINHOLE`
- **Shared for all images:** zapnuté
- **Parameters from EXIF:** zapnuté
- ostatné nastavenia som nechal predvolené

Tým sa extrahovali črty zo všetkých fotografií.

---

## 6. COLMAP – Feature matching
Ďalej som spustil:

```text
Processing → Feature matching → Exhaustive
```

Použil som predvolené nastavenia. Matching prebehol úspešne bez chýb.

---

## 7. COLMAP – Reconstruction
Potom som spustil:

```text
Reconstruction → Start reconstruction
```

COLMAP zaregistroval väčšinu snímok a vytvoril sparse rekonštrukciu scény.

Po dokončení som exportoval model do:

```bash
/opt/gaussian-splatting/data/auticko_3/distorted/sparse/0
```

V tomto priečinku vznikli súbory:

- `cameras.bin`
- `images.bin`
- `points3D.bin`
- `project.ini`

---

## 8. Undistortion a konverzia datasetu
Po úspešnom exporte sparse modelu som spustil:

```bash
python3 convert.py -s data/auticko_3
```

Tento krok pripravil dataset pre 3D Gaussian Splatting a vytvoril undistorted verziu dát.

---

## 9. Tréning 3D Gaussian Splatting
Následne som spustil tréning:

```bash
python3 train.py -s data/auticko_3
```

Tréning bežal na **GPU (RTX 4060)** a prebehol na:

- **7000 iterácií** – prvý checkpoint
- **30000 iterácií** – finálny checkpoint

Výsledky tréningu:

- **PSNR po 7000 iteráciách:** `26.74098815917969`
- **PSNR po 30000 iteráciách:** `29.796587371826174`

Checkpointy sa uložili do:

```bash
/opt/gaussian-splatting/output/512a32f1-c/point_cloud/iteration_7000
/opt/gaussian-splatting/output/512a32f1-c/point_cloud/iteration_30000
```

---

## 10. Vizualizácia výsledku
Na vizualizáciu som použil viewer:

```bash
SIBR_gaussianViewer_app -m /opt/gaussian-splatting/output/512a32f1-c
```

Vo vieweri bolo možné pozerať rekonštruované autíčko z rôznych uhlov a porovnať priebeh medzi checkpointmi.

---

## 11. Poznámky k výsledku
Výsledná rekonštrukcia autíčka bola úspešná. Objekt bol dobre rozpoznateľný, aj keď pozadie (textúrovaný tmavý podklad) malo výrazný vplyv na vzhľad rekonštrukcie.

Za hlavné faktory kvality považujem:

- dostatočný počet snímok,
- rozumný downscale obrázkov,
- správne nastavenie `SIMPLE_PINHOLE` a zdieľanej kamery v COLMAPe,
- tréning až do 30000 iterácií.

---

## 12. Súbory, ktoré chcem ponechať vo verziovaní
V repozitári chcem ponechať hlavne:

- priečinok `Auticko_model`
- `PVSO_zad_4.pdf`
- `trening.png`
- `trening_viac.png`
- `Postup.md`

Na tento účel je pripravený aj `.gitignore`.
