#!/bin/bash
echo What is the commit message?

read message

git add . && git commit -am "$message" && git push
