Po vytvoreni pointclaudu v COLMAPE me isli rovno trenovat

Lebo pri tomto konkrétnom bicykel datasete už máš pripravené to, čo train.py potrebuje:

images
sparse
kamery
body scény

A train.py si to vie načítať priamo.
convert.py je určený hlavne pre tvoj vlastný dataset z fotiek, keď máš niečo ako: 
data/auticko_3/input

V tvojom postupe pri autíčku bolo presne toto:

fotky v input
sparse model z COLMAPu
potom python3 convert.py -s data/auticko_3
až potom train.py

Prečo pri bicykli stačí len train.py:
Lebo bicykel dataset už nie je „surový“ ako autíčko. Je už pripravený v štýle:
PVSO/Fotky_bicykel/bicycle/
 ├── images
 ├── sparse
 ├── images_2
 ├── images_4
 └── images_8
 
To znamená:

images = vstupné obrázky
sparse = kamery a rekonštrukcia
images_2/4/8 = downsample verzie

Čiže train.py má všetko a ide rovno.

A to si aj videl podľa výpisu:

čítal 194 kamier
načítal scénu
začal tréning

Keby dataset nebol pripravený správne, train.py by sa vôbec nerozbehol.

Teda stručne

Pri autíčku:

COLMAP -> convert.py -> train.py

Pri bicykli:

train.py

lebo bicykel už má pripravené dáta, takze: 
cd /opt/gaussian-splatting
python3 train.py -s PVSO/Fotky_bicykel/bicycle -m PVSO/Bike_Gaussian
