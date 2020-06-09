set -ex
NYC_TAXI="$HOME/nyc-taxi-binder"
if [ \! -d "$NYC_TAXI" ]; then
    mkdir "$NYC_TAXI"
fi
# 03 04 05 06
for n in 01 02
do
    if [ \! -f 'yellow_tripdata_2015-'$n'.csv.bz2' ]; then
        wget -nc "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2015-${n}.csv" -O  "$NYC_TAXI/yellow_tripdata_2015-${n}.csv"
        bzip2 "$NYC_TAXI/yellow_tripdata_2015-${n}.csv" &
    fi
done
