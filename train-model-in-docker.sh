#!/bin/sh
docker run -it --rm -p 17888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit python model.py