sudo docker-compose stop freqtrade;
sudo docker-compose up -d freqtrade;
sleep 30;
for n in {2..12};
do
        sudo docker-compose stop freqtrade$n;
        sudo docker-compose up -d freqtrade$n;
        sleep 30;
done