dirlist=$1

for curdir in $(cat $dirlist)
do
echo $curdir
python audio_processing.py --metadata_folder /bigdisk1/meta_silentcities/dbfiles/ --site $curdir --folder /bigdisk1/silentcities/$curdir --database /bigdisk1/database_pross.pkl --batch_size 128
python audio_processing.py --metadata_folder /bigdisk1/meta_silentcities/dbfiles/ --site $curdir --folder /bigdisk2/silentcities/$curdir --database /bigdisk1/database_pross.pkl --batch_size 128
done