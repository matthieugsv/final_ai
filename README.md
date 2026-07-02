# Lecture automatique de manometres (gauge_reader.py)

Script Python qui lit la valeur (0 a 1 MPa) affichee par l'aiguille de 3
manometres analogiques SMC a partir d'une photo, par traitement d'image
classique (OpenCV) :

- detection des 3 cadrans (transformee de Hough)
- segmentation par seuillage de l'aiguille
- regression lineaire angle -> valeur

## Usage

```
python gauge_reader.py chemin/vers/image.jpg [--out annotated.jpg]
```

Affiche les 3 valeurs et enregistre une image annotee (cadrans detectes +
angle de l'aiguille) pour verification visuelle.

## Contenu

- `gauge_reader.py` : script principal
- `dataset/` : photos d'exemple (3 capteurs par image)
- `exemple_resultats/` : images annotees generees + visuels de verification

## Limite connue

Chaque capteur est une unite physique montee separement et peut presenter
une legere rotation de montage differente. Le calibrage angle -> valeur est
partage entre les 3 capteurs (voir commentaires dans `gauge_reader.py`) ;
une erreur de quelques centiemes de MPa est possible sur un capteur dont le
montage differe notablement de la reference.
# final_ai
