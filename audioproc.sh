dirlist=$1

for curdir in $(cat $dirlist)
do
python audio_processing.py --metadata_folder /bigdisk2/meta_silentcities/site/ --site $curdir --folder /bigdisk1/silentcities/$curdir --database /bigdisk2/meta_silentcities/database_pross.pkl --tomp3 /bigdisk2/mp3_sl/ --batch_size 128
done
