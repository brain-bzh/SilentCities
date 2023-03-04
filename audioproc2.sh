dirlist=$1

for curdir in $(cat $dirlist)
do
rm -f /bigdisk2/meta_silentcities/site/${curdir}_metaloader.pkl
rm -f /bigdisk2/meta_silentcities/site/${curdir}_process.pkl
python audio_processing.py --metadata_folder /bigdisk2/meta_silentcities/site/ --site $curdir --folder /bigdisk2/silentcities/$curdir --database /bigdisk2/meta_silentcities/database_pross.pkl --batch_size 64 > process-site_${curdir}.txt
done
