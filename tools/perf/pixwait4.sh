log=run105-pix.txt; j=j105w.txt; exe=victory-amp13-diag
for i in $(seq 1 400); do
  sleep 9
  python -I joinlog.py $log $j 2>/dev/null
  if ! tasklist //FI "IMAGENAME eq $exe.exe" 2>/dev/null | grep -qi "$exe"; then echo EXITED-BEFORE; tail -2 $j | cut -c1-240; exit 0; fi
  last=$(grep -a "fps," $j | tail -4 | grep -o "[0-9]* draws/frame, dropped [0-9]*")
  ok=$(echo "$last" | awk '{d=$1; x=$4; if (d>=4000 && x<=600) n++} END{print n+0}')
  [ $((i % 10)) -eq 0 ] && echo "$(date +%H:%M:%S) $(echo "$last" | tail -1)"
  if [ "$ok" -ge 4 ]; then
    echo "TRIGGER at $(date +%H:%M:%S)"
    : > "<work>/profiles/pix-trigger.txt"
    for k in $(seq 1 100); do
      sleep 5
      if ! tasklist //FI "IMAGENAME eq $exe.exe" 2>/dev/null | grep -qi "$exe"; then echo "GAME EXITED after $((k*5)) s"; break; fi
      f=$(ls -t "<work>/profiles/"capture-*.wpix 2>/dev/null | head -1)
      if [ -n "$f" ] && [ $k -ge 8 ]; then age=$(( $(date +%s) - $(stat -c %Y "$f") )); if [ $age -ge 40 ]; then echo "FILE QUIET after $((k*5)) s: $(stat -c %s "$f") bytes"; break; fi; fi
    done
    python -I joinlog.py $log $j 2>/dev/null
    grep -a -i "PIX capture\|first-chance\|parked" $j | tail -4 | cut -c1-200
    ls -la "<work>/profiles/" | grep wpix
    exit 0
  fi
done
echo TIMEOUT
