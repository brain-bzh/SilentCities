dirlist=$1

for curdir in $(cat $dirlist)
do
python metadata_file.py --save_path /bigdisk1/meta_silentcities/dbfiles --folder /bigdisk1/silentcities/$curdir 
done
