for CURSITE in $(ls /Volumes/LaCie/Dataset_OHM_PYRENEES/OHM_PYRENEES_2020/)
do
	echo python Ecoacoustic_test.py --site $CURSITE --data_path /Volumes/LaCie/Dataset_OHM_PYRENEES/OHM_PYRENEES_2020/${CURSITE}/ --save_path /Volumes/LaCie/Dataset_OHM_PYRENEES/2020-results/
done

for CURSITE in $(ls /Volumes/LaCie/Dataset_OHM_PYRENEES/OHM_PYRENEES_2020/)
do
	echo python taggin_ohm.py --site $CURSITE --data_path /Volumes/LaCie/Dataset_OHM_PYRENEES/OHM_PYRENEES_2020/${CURSITE}/ --save_path /Volumes/LaCie/Dataset_OHM_PYRENEES/2020-results/
done