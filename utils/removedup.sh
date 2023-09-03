filelist=$1

for curfile in $(cat $filelist)
do
python remove_duplicates.py $curfile
done