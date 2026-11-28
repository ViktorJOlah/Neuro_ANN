#!/bin/bash
cd /home/user/npx
B=https://allen-brain-observatory.s3.us-west-2.amazonaws.com/visual-coding-neuropixels/ecephys-cache
one() { s=$1; [ -f ext/$s.npz ] && return; for i in 1 2 3; do curl -s -o s$s.nwb $B/session_$s/session_$s.nwb && break; sleep 10; done
  python code/extract.py $s s$s.nwb ext/$s.npz >> extract.log 2>&1; rm -f s$s.nwb; }
export -f one; export B
cat sessions_todo.txt | xargs -P 3 -I{} bash -c 'one {}'
echo FETCH_DONE >> extract.log
