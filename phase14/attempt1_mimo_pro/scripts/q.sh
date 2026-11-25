#!/bin/bash
# quick tail/grep helper: q.sh tail <file> <n> | q.sh grep <file> <pattern>
case "$1" in
  tail) tail -n "${3:-5}" "/home/user/wsp14/work/$2" ;;
  grep) grep -n -- "${3}" "/home/user/wsp14/work/$2" ;;
  count) grep -c -- "${3}" "/home/user/wsp14/work/$2" ;;
  cat) cat "/home/user/wsp14/work/$2" ;;
esac
