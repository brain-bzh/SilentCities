filelist=$1

for curfile in $(cat $filelist)
do
python remove_dup_db.py $curfile
done