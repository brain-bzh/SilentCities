dirlist=$1

for curdir in $(cat $dirlist)
do
python convert.py --metadata_folder /bigdisk1/meta_silentcities/dbfiles --site $curdir  --results_folder /bigdisk1/meta_silentcities/AcousticMeasurements_renamed_withspeech/ --database /bigdisk1/database_pross.pkl --batch_size 16 --toflac /bigdisk1/flac --destdir /nasbrain/datasets/silentcities > log-convert/report${curdir}
# create new variable with the name of the current directory without the first character
curdir2=${curdir:1}
mv /bigdisk1/flac/$curdir2 /nasbrain/datasets/silentcities
done