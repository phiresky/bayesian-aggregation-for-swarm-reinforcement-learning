#!/bin/bash
jq '.argv[]' "$1" | tr '\n' ' '