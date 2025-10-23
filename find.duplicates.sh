#!/bin/bash

# Navigate to the target directory
cd /net/birdstore/Active_Atlas_Data/data_root/pipeline_data/DK79/preps/C1/thumbnail_cleaned || exit 1

# Calculate MD5 hash for each file and store in a temporary file
find . -maxdepth 1 -type f -print0 | xargs -0 md5sum > /tmp/checksums.txt

# Sort the checksums and identify duplicates
sort /tmp/checksums.txt | uniq -w 32 --all-repeated=separate | awk '
BEGIN {
    prev_hash = ""
    file_list = ""
}
{
    current_hash = substr($0, 1, 32)
    current_file = substr($0, 35)

    if (current_hash == prev_hash) {
        file_list = file_list "\n" current_file
    } else {
        if (file_list != "") {
            print "Duplicate files with hash " prev_hash ":"
            print file_list
            print ""
        }
        prev_hash = current_hash
        file_list = current_file
    }
}
END {
    if (file_list != "") {
        print "Duplicate files with hash " prev_hash ":"
        print file_list
        print ""
    }
}'

# Clean up the temporary file
rm /tmp/checksums.txt
