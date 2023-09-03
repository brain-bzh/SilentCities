filelist=$1

for curfile in $(cat $filelist)
do
python clean_empty_files.py $curfile
done