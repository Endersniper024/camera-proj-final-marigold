#!/usr/bin/env bash
set -u
URL="https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/eth3d/eth3d.tar"
TOTAL=49429913600
N=16
DIR=/root/autodl-tmp/eth3d
mkdir -p "$DIR/_chunks"
cd "$DIR/_chunks"
BASE=$(( (TOTAL + N - 1) / N ))

dl_chunk() {
  local i=$1
  local start=$(( i * BASE ))
  local end=$(( start + BASE - 1 ))
  [ $end -ge $TOTAL ] && end=$(( TOTAL - 1 ))
  local out="c$(printf %02d $i).part"
  local target=$(( end - start + 1 ))
  local tries=0
  while true; do
    local have=0
    [ -f "$out" ] && have=$(stat -c%s "$out")
    [ $have -ge $target ] && break
    local rstart=$(( start + have ))
    curl -s --retry 8 --retry-delay 4 -r ${rstart}-${end} "$URL" >> "$out"
    have=$(stat -c%s "$out")
    [ $have -ge $target ] && break
    tries=$((tries+1))
    [ $tries -gt 200 ] && { echo "chunk $i GAVE UP at $have/$target"; return 1; }
    sleep 3
  done
  echo "chunk $i done ($have bytes)"
}
export -f dl_chunk; export URL TOTAL N BASE

echo "START $(date)  base_chunk=$BASE"
for i in $(seq 0 $((N-1))); do dl_chunk $i & done
wait
echo "ALL CHUNKS FETCHED $(date)"

SUM=0
for i in $(seq 0 $((N-1))); do
  f="c$(printf %02d $i).part"; s=$(stat -c%s "$f" 2>/dev/null || echo 0); SUM=$((SUM+s))
done
echo "concatenated-bytes-check: $SUM vs $TOTAL"
if [ "$SUM" -ne "$TOTAL" ]; then echo "SIZE MISMATCH - abort assemble"; echo "ETH3D_DL FAIL"; exit 1; fi

echo "assembling..."
cat $(for i in $(seq 0 $((N-1))); do printf "c%02d.part " $i; done) > "$DIR/eth3d.tar"
FIN=$(stat -c%s "$DIR/eth3d.tar")
echo "final eth3d.tar = $FIN bytes"
if [ "$FIN" -eq "$TOTAL" ]; then
  rm -rf "$DIR/_chunks"
  echo "ETH3D_DL OK"
else
  echo "ETH3D_DL FAIL (final size mismatch)"
fi
