#!/bin/bash

# Delete models:
find . -iname 'VGG19*' -type f | xargs rm
find . -iname 'ResNet50*' -type f | xargs rm

# Delete results:
find . -iname 'results*' -type f | xargs rm

# Delete __pycache__:
find . -iname '*pycache*' | xargs "rm -r"



