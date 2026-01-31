#!/bin/bash

while [[ $# -gt 0 ]]; do
    case $1 in
        --time_stamp)
            time_stamp="$2"
            shift 2
            ;;
        --time_stamp=*)
            time_stamp="${1#*=}"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --time_stamp <time_stamp>"
            exit 1
            ;;
    esac
done

if [ -z "$time_stamp" ]; then
    echo "Usage: $0 --time_stamp <time_stamp>"
    exit 1
fi

ffmpeg -i "logs/screen_recording_${time_stamp}.mp4" \
  -c:v hevc_nvenc \
  -preset p7 \
  -cq:v 32 \
  -tune hq \
  -c:a aac -b:a 64k \
  "logs/screen_recording_${time_stamp}_h265.mp4"

# NVIDIA's quality-focused preset p7
# Higher CQ = lower quality/smaller file (35-38 for screen)
# High quality tuning hq