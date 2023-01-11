aocl help > /dev/null || source init_env.sh

date

bin/host /share/ScratchGeneral/kisliy/chaining_data/in-human_new-20k.txt > /dev/null
bin/host /share/ScratchGeneral/kisliy/chaining_data/in-human_new-30k.txt > /dev/null
bin/host /share/ScratchGeneral/kisliy/chaining_data/in-human_new-40k.txt > /dev/null
bin/host /share/ScratchGeneral/kisliy/chaining_data/in-human_new-500k.txt > /dev/null

date