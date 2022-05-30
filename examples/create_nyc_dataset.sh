#!/bin/sh

set -x

if [ \! -d examples ]; then
    echo 'Should call the script from the main progressivis directory (with example/ as subdirectory)'
    exit 1
fi
if [ \! -d ../nyc-taxi ]; then
    mkdir ../nyc-taxi
fi

cd ../nyc-taxi
for n in 01 02 03 04 05 06 07 08 09 10 11 12
do
    if [ \! -f 'green_tripdata_2015-'$n'.csv.bz2' ]; then
        wget -nc 'https://nyc-tlc.s3.amazonaws.com/csv_backup/green_tripdata_2015-'$n'.csv'
        bzip2 'green_tripdata_2015-'$n'.csv' &
    fi
done

for n in 01 02 03 04 05 06 07 08 09 10 11 12
do
    if [ \! -f 'yellow_tripdata_2015-'$n'.csv.bz2' ]; then
        wget -nc 'https://nyc-tlc.s3.amazonaws.com/csv_backup/yellow_tripdata_2015-'$n'.csv'
        bzip2 'yellow_tripdata_2015-'$n'.csv' &
    fi
done

for n in apr may jun jul aug sep
do
    if [ \! -f 'uber-raw-data-'$n'14.csv.bz2' ]; then
        wget -nc 'https://raw.githubusercontent.com/fivethirtyeight/uber-tlc-foil-response/master/uber-trip-data/uber-raw-data-'$n'14.csv'
        bzip2 'uber-raw-data-'$n'14.csv' &
    fi
done
