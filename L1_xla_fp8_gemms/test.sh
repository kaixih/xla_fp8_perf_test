
if [[ -z "$1" ]]; then
  echo "Checking model with no repeated layer:"
  bash base.sh y 1 False FWD 4
  bash base.sh n 1 False BWD 8
  echo "Checking model with repeated layer:"
  bash base.sh y 1 True FWD 7
  bash base.sh n 1 True BWD 8
fi

if [ "$1" == "verbose" ]; then
  echo "Checking model with no repeated layer (verbose):"
  bash base.sh y 1 False FWD 1 QKV
  bash base.sh n 1 False FWD 1 POST
  bash base.sh n 1 False FWD 1 FF1
  bash base.sh n 1 False FWD 1 FF2
  bash base.sh n 1 False BWD 2 QKV
  bash base.sh n 1 False BWD 2 POST
  bash base.sh n 1 False BWD 2 FF1
  bash base.sh n 1 False BWD 2 FF2
  echo "Checking model with repeated layer (verbose):"
  bash base.sh y 1 True FWD 2 QKV
  bash base.sh n 1 True FWD 2 POST
  bash base.sh n 1 True FWD 2 FF1
  bash base.sh n 1 True FWD 1 FF2
  bash base.sh n 1 True BWD 2 QKV
  bash base.sh n 1 True BWD 2 POST
  bash base.sh n 1 True BWD 2 FF1
  bash base.sh n 1 True BWD 2 FF2
fi


