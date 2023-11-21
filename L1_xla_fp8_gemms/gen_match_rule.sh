COUNT=$1
PROP_DIRECTION=$2
OP_NAME=$3

if [[ "$PROP_DIRECTION" == "FWD" ]]; then
  PRECISION=DEFAULT
elif [[ "$PROP_DIRECTION" == "BWD" ]]; then
  PRECISION=HIGHEST
else
  echo FAILED: Unsupported PROP_DIRECTION: $PROP_DIRECTION
  exit 1
fi

RULE=\"__cublas\$lt\$matmul\$f8\"{{.*}}$OP_NAME{{.*}}\"operand_precision\":[\"$PRECISION\",\"$PRECISION\"]
cat > match_rules_gen.ll <<EOF
; CHECK-COUNT-$COUNT: $RULE
; CHECK-NOT: $RULE
EOF

