dirlist=$1

for curdir in $(cat $dirlist)
do
echo "#!/bin/bash" > $curdir.sh
echo "#SBATCH --job-name=process-site_${curdir}_%j" >> $curdir.sh
echo "#SBATCH --output=process-site_${curdir}_%j.txt" >> $curdir.sh
echo "#SBATCH --gres=gpu:gtx1060:1" >> $curdir.sh

echo "echo ""Cleaning previous files"" " >> $curdir.sh
echo "rm -Rf /users/local/tmpsl/" >> $curdir.sh
echo "rm -f /users/local/bigdisk2/meta_silentcities/site/${curdir}_metaloader.pkl " >> $curdir.sh
echo "rm -f /users/local/bigdisk2/meta_silentcities/site/${curdir}_process.pkl " >> $curdir.sh
echo "echo ""Creating dirs"" " >> $curdir.sh
echo "mkdir -p /users/local/tmpsl/ " >> $curdir.sh
echo "mkdir -p /users/local/tmpsl/wav " >> $curdir.sh
echo "#mkdir -p /users/local/tmpsl/mp3 " >> $curdir.sh
echo "mkdir -p /users/local/tmpsl/meta " >> $curdir.sh
echo "echo ""Copying data..."" " >> $curdir.sh
echo "cp -R /users/local/bigdisk1/silentcities/${curdir}/ /users/local/tmpsl/wav/ " >> $curdir.sh
echo "cp /users/local/bigdisk2/meta_silentcities/site/${curdir}* /users/local/tmpsl/meta/ " >> $curdir.sh

echo "echo ""Processing..."" " >> $curdir.sh
echo "source /opt/campux/virtualenv/deeplearning-u20-rtx-3080/bin/activate" >> $curdir.sh
echo "python audio_processing.py --metadata_folder /users/local/tmpsl/meta/ --site $curdir --folder /users/local/tmpsl/wav --database /users/local/bigdisk2/meta_silentcities/database_pross.pkl --batch_size 64" >> $curdir.sh

echo "echo ""Copying results..."" " >> $curdir.sh
echo "cp /users/local/tmpsl/meta/* /users/local/bigdisk2/meta_silentcities/site/" >> $curdir.sh
echo "echo ""Cleaning..."" " >> $curdir.sh
echo "rm -Rf /users/local/tmpsl/" >> $curdir.sh
echo "echo ""Finished! "" " >> $curdir.sh
done