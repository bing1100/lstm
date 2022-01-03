python lstmcustom.py 2>&1 |& tee lstm_imdb_no_update1.txt
python lstmcustom.py 2>&1 |& tee lstm_imdb_no_update2.txt
python lstmcustom.py 2>&1 |& tee lstm_imdb_no_update3.txt
python lstmcustom.py 2>&1 |& tee lstm_imdb_no_update4.txt
python lstmcustom.py 2>&1 |& tee lstm_imdb_no_update5.txt

python lstmcustom.py --update 2>&1 |& tee lstm_imdb_update1.txt
python lstmcustom.py --update 2>&1 |& tee lstm_imdb_update2.txt
python lstmcustom.py --update 2>&1 |& tee lstm_imdb_update3.txt
python lstmcustom.py --update 2>&1 |& tee lstm_imdb_update4.txt
python lstmcustom.py --update 2>&1 |& tee lstm_imdb_update5.txt