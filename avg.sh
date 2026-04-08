#!/bin/bash

if [ $# -lt 1 ]; then
  echo "Usage: $0 <filename>"
  exit 1
fi

awk '
{
  if ($1 ~ /^-?[0-9]+(\.[0-9]+)?$/) {
    sum += $1
    count++
  }
}
END {
  if (count > 0)
    print "Average:", sum / count
  else
    print "No valid numbers found."
}
' "$1"
