#!/bin/bash

# Delete models:
find . -iname 'VGG19*' -type f | xargs rm
find . -iname 'ResNet50*' -type f | xargs rm

# Delete results:
find . -iname 'results*' -type f | xargs rm

# Delete python cache:
find . -iname '*pycache*' | xargs rm -r
find . -iname '*.pyc' | xargs rm


