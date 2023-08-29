parallel-ssh -A -i -h hosts tmux new -d -s nfa-homedir-persistent 'krenew -K 60 watch ls'
parallel-ssh -A -i -h hosts "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-513.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-513.imta.fr:/bigdisk2 /users/local/bigdisk2"
#squeue -o %.18i %.9P %.20j %.8u %.2t %.10M %.6D %R
#TEST COMMAND
#python audio_processing.py --metadata_folder /users3/local/bigdisk2/meta_silentcities/TEST/ --site 0000 --folder /users3/local/bigdisk1/silentcities/0000/ --database /users3/local/bigdisk2/meta_silentcities/TEST/database_pross.pkl --batch_size 64












