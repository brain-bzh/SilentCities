#for CURSITE in $(ls /Volumes/LaCie/Dataset_OHM_PYRENEES/OHM_PYRENEES_2020/)
#do
#	python Ecoacoustic_test.py --site $CURSITE --data_path /Volumes/LaCie/Dataset_OHM_PYRENEES/OHM_PYRENEES_2020/${CURSITE}/ --save_path /Volumes/LaCie/Dataset_OHM_PYRENEES/2020-results/
#done

#for CURSITE in $(ls /Volumes/LaCie/Dataset_OHM_PYRENEES/OHM_PYRENEES_2020/)
#do
#	python tagging_ohm.py --site $CURSITE --data_path /Volumes/LaCie/Dataset_OHM_PYRENEES/OHM_PYRENEES_2020/${CURSITE}/ --save_path /Volumes/LaCie/Dataset_OHM_PYRENEES/2020-results/
#done

for CURFILE in $(ls /media/nfarrugi/datapal/ohm/2020-results/2020-results/tagging_site_*)
do
	python Extract_timeranges-OHM-Tagging.py --input ${CURFILE} --save_path /media/nfarrugi/datapal/ohm/2020-results/2020-results/
done