parallel-ssh -A -i -h /home/nfarrugi/hosts.txt tmux new -d -s nfa-homedir-persistent 'krenew -K 60 watch ls'
parallel-ssh -A -i -h /home/nfarrugi/hosts.txt "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
#squeue -o %.18i %.9P %.20j %.8u %.2t %.10M %.6D %R













