parallel-ssh -A -i -h hosts.txt tmux new -d -s nfa-homedir-persistent 'krenew -K 60 watch ls'
ssh pc-elec-385.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-386.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-387.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-388.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-389.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-390.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-391.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-392.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-393.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-394.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-395.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"
ssh pc-elec-396.priv.enst-bretagne.fr "mkdir -p /users/local/bigdisk1 ; mkdir -p /users/local/bigdisk2 ; sshfs sl-tp-br-008.imta.fr:/bigdisk1 /users/local/bigdisk1 ; sshfs sl-tp-br-008.imta.fr:/bigdisk2 /users/local/bigdisk2"

#squeue -o %.18i %.9P %.20j %.8u %.2t %.10M %.6D %R













