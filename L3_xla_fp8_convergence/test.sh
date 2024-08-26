
if [ -z "$1" ]; then
  echo Running cudnn flash attention
  bash base.sh GPT5B XLA fp8 FA 8
elif [ "$1" = "debug" ]; then
  echo Running with no cudnn attention
  bash base.sh GPT5B XLA fp8 NA 8
fi

