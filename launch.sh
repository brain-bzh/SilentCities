dirlist=$1

for curdir in $(cat $dirlist)
do
python metadata_file.py --save_path /bigdisk2/meta_silentcities/site --folder /bigdisk1/silentcities/$curdir 
done
