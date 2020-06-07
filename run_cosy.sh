for i in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15
do
name="cosy${i}"
echo "Connecting to $name"
ssh "$name" /bin/bash << 'EOT'
cd Dokumente
source ppo-env/bin/activate
cd master-thesis/project
screen -d -m python -m source.main --file
EOT
done
